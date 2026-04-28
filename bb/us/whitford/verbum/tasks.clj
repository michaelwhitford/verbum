(ns us.whitford.verbum.tasks
  "Clojure → lambda notation converter + BIOS flash data generator.

   clj2lambda: Mechanically converts Clojure source into lambda calculus
   notation for use as VSM training data. Covers ~96% of Clojure: defn,
   fn, let, if/when/cond, threading macros, destructuring, loop/recur,
   and all pure clojure.core higher-order functions.
   Skips: Java interop, complex macros, eval/resolve, mutable state.

   gen-bios: Generate BIOS flash training data (math + clojure.core).
   Delegates to us.whitford.verbum.bios.

   Architecture: read-string → walk → emit lambda text.
   No rewrite-clj needed — Clojure is homoiconic, the reader IS
   the parser."
  (:require [babashka.cli :as cli]
            [babashka.fs :as fs]
            [cheshire.core :as json]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.walk :as walk]))

;; ═══════════════════════════════════════════════════════════════
;; Lambda emission
;; ═══════════════════════════════════════════════════════════════

(defn emit-lambda
  "Convert a transformed form to lambda notation string."
  [form]
  (cond
    (nil? form)    "nil"
    (symbol? form) (str form)
    (keyword? form) (str form)
    (string? form) (pr-str form)
    (number? form) (str form)
    (boolean? form) (str form)
    (char? form) (pr-str form)
    (instance? java.util.regex.Pattern form) (str "(re " (pr-str (str form)) ")")

    ;; λx. body
    (and (seq? form) (= 'λ (first form)))
    (let [[_ params body] form]
      (if (sequential? params)
        (str (str/join "" (map #(str "λ" % ". ") params)) (emit-lambda body))
        (str "λ" params ". " (emit-lambda body))))

    ;; (apply f args...)
    (seq? form)
    (str "(" (str/join " " (map emit-lambda form)) ")")

    ;; [a b c]
    (vector? form)
    (str "[" (str/join " " (map emit-lambda form)) "]")

    ;; {:k v}
    (map? form)
    (str "{" (str/join " " (map (fn [[k v]] (str (emit-lambda k) " " (emit-lambda v))) form)) "}")

    ;; #{a b}
    (set? form)
    (str "#{" (str/join " " (map emit-lambda form)) "}")

    :else (str form)))

;; ═══════════════════════════════════════════════════════════════
;; Clojure → lambda transformation
;; ═══════════════════════════════════════════════════════════════

(declare transform)

(defn- transform-bindings
  "Transform let/loop bindings [x 1 y 2] into nested lambda applications.
   (let [x 1 y 2] body) → ((λx. ((λy. body) 2)) 1)"
  [bindings body]
  (if (empty? bindings)
    (transform body)
    (let [[sym val & rest-bindings] bindings]
      (list (list 'λ (transform sym) (transform-bindings (vec rest-bindings) body))
            (transform val)))))

(defn- transform-fn-params
  "Transform [x y z] into nested lambda: λx. λy. λz. body"
  [params body-forms]
  (let [body (if (= 1 (count body-forms))
               (transform (first body-forms))
               (cons 'do (map transform body-forms)))]
    (if (empty? params)
      (list 'λ '_ body)  ; (fn [] body) → λ_. body
      (list 'λ params body))))

(defn- transform-cond-pairs
  "Transform cond pairs into nested if expressions."
  [pairs]
  (if (empty? pairs)
    'nil
    (let [[test expr & rest-pairs] pairs]
      (if (= :else test)
        (transform expr)
        (list 'if (transform test) (transform expr)
              (transform-cond-pairs rest-pairs))))))

(defn- threading-first
  "Expand -> form: (-> x (f a) (g b)) → (g (f x a) b)"
  [x & forms]
  (reduce (fn [acc form]
            (if (seq? form)
              (let [[f & args] form]
                (apply list f acc args))
              (list form acc)))
          x forms))

(defn- threading-last
  "Expand ->> form: (->> x (f a) (g b)) → (g a (f a x))"
  [x & forms]
  (reduce (fn [acc form]
            (if (seq? form)
              (let [[f & args] form]
                (apply list f (concat args [acc])))
              (list form acc)))
          x forms))

(defn transform
  "Transform a Clojure form into lambda notation.

   Dispatch on special forms and macros. Everything else passes
   through as function application."
  [form]
  (cond
    ;; Atoms pass through
    (nil? form) nil
    (symbol? form) form
    (keyword? form) form
    (string? form) form
    (number? form) form
    (boolean? form) form
    (char? form) form

    ;; Collections — transform contents
    (vector? form) (mapv transform form)
    (map? form) (into {} (map (fn [[k v]] [(transform k) (transform v)]) form))
    (set? form) (into #{} (map transform form))

    ;; S-expressions — the interesting part
    (seq? form)
    (let [[head & args] form]
      (case head
        ;; ── Definitions ──────────────────────────────────────
        ;; (defn foo [x y] body) → (def foo (λx. λy. body))
        defn
        (let [[name params & body] args]
          (list 'def name (transform-fn-params params body)))

        defn-
        (let [[name params & body] args]
          (list 'def name (transform-fn-params params body)))

        ;; (def x 42) → (def x 42)
        def
        (let [[name val] args]
          (list 'def name (transform val)))

        ;; ── Lambda ───────────────────────────────────────────
        ;; (fn [x y] body) → λx. λy. body
        fn
        (let [;; Handle (fn name [x] body) and (fn [x] body)
              [params & body] (if (symbol? (first args))
                                (rest args)
                                args)]
          (transform-fn-params params body))

        ;; ── Binding ──────────────────────────────────────────
        ;; (let [x 1 y 2] body) → ((λx. ((λy. body) 2)) 1)
        let
        (let [[bindings & body] args]
          (transform-bindings bindings
                              (if (= 1 (count body))
                                (first body)
                                (cons 'do body))))

        ;; ── Conditionals ─────────────────────────────────────
        if
        (let [[test then else] args]
          (list 'if (transform test) (transform then) (transform else)))

        when
        (let [[test & body] args]
          (list 'if (transform test)
                (if (= 1 (count body))
                  (transform (first body))
                  (cons 'do (map transform body)))
                nil))

        cond
        (transform-cond-pairs args)

        case
        (let [[expr & clauses] args]
          ;; Simplify: case → nested if with =
          (let [pairs (partition-all 2 clauses)
                has-default? (odd? (count clauses))
                default (when has-default? (last clauses))
                test-pairs (if has-default? (butlast pairs) pairs)]
            (reduce (fn [else [test-val then]]
                      (list 'if (list '= (transform expr) test-val)
                            (transform then) else))
                    (if has-default? (transform default) nil)
                    (reverse test-pairs))))

        ;; ── Threading ────────────────────────────────────────
        ->  (transform (apply threading-first args))
        ->> (transform (apply threading-last args))

        ;; ── Loops ────────────────────────────────────────────
        ;; (loop [x 0] (if (< x 10) (recur (inc x)) x))
        ;; → (fix (λloop. λx. (if (< x 10) (loop (inc x)) x)) 0)
        loop
        (let [[bindings & body] args
              params (take-nth 2 bindings)
              inits (take-nth 2 (rest bindings))
              body-form (if (= 1 (count body))
                          (first body)
                          (cons 'do body))]
          (apply list 'fix
                 (list 'λ (vec (cons 'recur params))
                       (transform body-form))
                 (map transform inits)))

        recur
        (apply list 'recur (map transform args))

        ;; ── Sequences / do ───────────────────────────────────
        do
        (if (= 1 (count args))
          (transform (first args))
          (cons 'do (map transform args)))

        ;; ── Interop (opaque) ─────────────────────────────────
        ;; Mark Java interop as opaque — the 4%
        new   (apply list 'new! args)
        throw (list 'throw! (transform (first args)))

        ;; ── Quote ────────────────────────────────────────────
        quote form  ; preserve quoted forms as-is

        ;; ── Default: function application ────────────────────
        (apply list (transform head) (map transform args))))

    :else form))

;; ═══════════════════════════════════════════════════════════════
;; File processing
;; ═══════════════════════════════════════════════════════════════

(defn read-forms
  "Read all forms from a Clojure source string.
   Returns a seq of forms, skipping read errors."
  [source]
  (let [reader (java.io.PushbackReader. (java.io.StringReader. source))]
    (loop [forms []]
      (let [form (try (edn/read {:eof ::eof} reader)
                      (catch Exception e
                        (binding [*out* *err*]
                          (println "  SKIP (read error):" (.getMessage e)))
                        ::skip))]
        (cond
          (= ::eof form)  forms
          (= ::skip form) forms  ; stop on first error, return what we have
          :else            (recur (conj forms form)))))))

(defn convert-source
  "Convert a Clojure source string to a seq of lambda notation strings.
   Each top-level form becomes one entry."
  [source]
  (->> (read-forms source)
       (map (fn [form]
              (try
                {:status :ok
                 :clojure (pr-str form)
                 :lambda  (emit-lambda (transform form))}
                (catch Exception e
                  {:status :error
                   :clojure (pr-str form)
                   :error (.getMessage e)}))))
       (filter some?)))

(defn convert-file
  "Convert a single .clj file. Returns seq of conversion records."
  [path]
  (let [source (slurp (str path))]
    (map #(assoc % :source-file (str path))
         (convert-source source))))

;; ═══════════════════════════════════════════════════════════════
;; CLI
;; ═══════════════════════════════════════════════════════════════

(def cli-spec
  {:input  {:desc    "Input: .clj file, directory, or - for stdin"
            :alias   :i
            :default "-"}
   :output {:desc    "Output JSONL file (default: stdout)"
            :alias   :o
            :default "-"}
   :recursive {:desc    "Recursively find .clj files in directory"
               :alias   :r
               :coerce  :boolean
               :default true}
   :help   {:desc   "Show help"
            :alias  :h
            :coerce :boolean}})

(defn- find-clj-files
  "Find all .clj files under a directory."
  [dir]
  (->> (fs/glob dir "**.clj")
       (map str)
       (sort)))

(defn- write-jsonl
  "Write records as JSONL to writer."
  [writer records]
  (doseq [rec records]
    (.write writer (json/generate-string rec))
    (.write writer "\n")))

(defn clj2lambda
  "Entry point for the clj2lambda task."
  [& _args]
  (let [opts (cli/parse-opts *command-line-args* {:spec cli-spec})]
    (if (:help opts)
      (do
        (println "clj2lambda — Convert Clojure source to lambda notation")
        (println)
        (println "Usage:")
        (println "  bb clj2lambda -i src/my/ns.clj           # single file")
        (println "  bb clj2lambda -i src/ -o train.jsonl      # directory")
        (println "  cat foo.clj | bb clj2lambda               # stdin")
        (println)
        (println "Options:")
        (println (cli/format-opts {:spec cli-spec})))
      (let [input  (:input opts)
            output (:output opts)
            files  (cond
                     (= "-" input)       nil  ; stdin mode
                     (fs/directory? input) (find-clj-files input)
                     (fs/exists? input)   [(str input)]
                     :else (do (binding [*out* *err*]
                                 (println "Error: input not found:" input))
                               (System/exit 1)))
            records (if files
                      (mapcat (fn [f]
                                (binding [*out* *err*]
                                  (println "  Converting:" f))
                                (convert-file f))
                              files)
                      ;; stdin mode
                      (convert-source (slurp *in*)))
            ok-count    (count (filter #(= :ok (:status %)) records))
            error-count (count (filter #(= :error (:status %)) records))]
        (if (= "-" output)
          (write-jsonl *out* records)
          (with-open [w (io/writer output)]
            (write-jsonl w records)))
        (binding [*out* *err*]
          (println (str "Done: " ok-count " converted, " error-count " errors")))))))

;; ═══════════════════════════════════════════════════════════════
;; gen-bios — thin wrapper over bios.clj
;; ═══════════════════════════════════════════════════════════════

(def gen-bios-spec
  {:count {:desc    "Number of examples to generate"
           :alias   :n
           :coerce  :long
           :default 2560000}
   :seed  {:desc    "Random seed"
           :alias   :s
           :coerce  :long
           :default 42}
   :help  {:desc   "Show help"
           :alias  :h
           :coerce :boolean}})

(defn gen-bios
  "Entry point for the gen-bios task.
   Generates BIOS flash training data to stdout (one example per line).
   Stats printed to stderr."
  [& _args]
  (let [opts (cli/parse-opts *command-line-args* {:spec gen-bios-spec})]
    (if (:help opts)
      (do
        (println "gen-bios — Generate BIOS flash training data")
        (println)
        (println "Usage:")
        (println "  bb gen-bios                              # default 2.56M examples")
        (println "  bb gen-bios --count 1000 --seed 42       # small test run")
        (println "  bb gen-bios > bios_examples.txt           # save to file")
        (println)
        (println "Options:")
        (println (cli/format-opts {:spec gen-bios-spec})))
      (do
        (require 'us.whitford.verbum.bios)
        ((resolve 'us.whitford.verbum.bios/run)
         {:count (:count opts)
          :seed  (:seed opts)})))))
