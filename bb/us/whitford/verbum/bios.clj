(ns us.whitford.verbum.bios
  "BIOS flash training data generator.

   Generates math + clojure.core training examples with eval-verified
   results. Every expression is evaluated in babashka — no approximations,
   no hand-coded templates. Babashka IS the ground truth.

   Single notation per example forces computation every time:
     raw:    347 + 289 = 636
     sexpr:  (+ 347 289) → 636
     lambda: (λx. λy. (+ x y) 347 289) → 636

   Output: plain text, one example per line. Fed to Python packer for
   Qwen3 BBPE tokenization + .npy shard packing."
  (:require [clojure.string :as str]
            [us.whitford.verbum.tasks :as tasks]))

;; ═══════════════════════════════════════════════════════════════
;; Random input generators
;; ═══════════════════════════════════════════════════════════════

(def ^:dynamic *rng* (java.util.Random. 42))

(defn rand-int*
  "Random int in [lo, hi] inclusive."
  ([hi] (.nextInt *rng* (inc hi)))
  ([lo hi] (+ lo (.nextInt *rng* (- (inc hi) lo)))))

(defn rand-pos
  "Random positive int [1, hi]."
  [hi] (rand-int* 1 hi))

(defn rand-digits
  "Random int with 1-4 digits, biased toward small."
  []
  (let [d (rand-int* 0 9)]
    (cond
      (< d 4) (rand-int* 0 9)       ; 40% single digit
      (< d 7) (rand-int* 0 99)      ; 30% two digit
      (< d 9) (rand-int* 0 999)     ; 20% three digit
      :else   (rand-int* 0 9999)))) ; 10% four digit

(defn rand-signed
  "Random signed int, biased small."
  []
  (let [v (rand-digits)]
    (if (< (.nextDouble *rng*) 0.3) (- v) v)))

(defn rand-bool [] (< (.nextDouble *rng*) 0.5))

(defn rand-choice [coll] (nth coll (.nextInt *rng* (count coll))))

(defn rand-int-list
  "Random list of ints, length [min-n, max-n]."
  [min-n max-n]
  (let [n (rand-int* min-n max-n)]
    (vec (repeatedly n rand-digits))))

(defn rand-signed-list
  [min-n max-n]
  (let [n (rand-int* min-n max-n)]
    (vec (repeatedly n rand-signed))))

(defn rand-small-list
  "Small positive ints for mul-safe operations."
  [min-n max-n]
  (let [n (rand-int* min-n max-n)]
    (vec (repeatedly n #(rand-int* 1 15)))))

;; ═══════════════════════════════════════════════════════════════
;; Result formatting — canonical string representation
;; ═══════════════════════════════════════════════════════════════

(defn fmt-result
  "Format a Clojure value as canonical training string.
   Seqs → vector notation. Maps sorted by key."
  [v]
  (cond
    (nil? v) "nil"
    (boolean? v) (str v)
    (number? v) (str v)
    (string? v) (pr-str v)
    (keyword? v) (str v)
    (symbol? v) (str v)
    (set? v) (str "#{" (str/join " " (map fmt-result (sort v))) "}")
    (map? v) (str "{" (str/join " " (map (fn [[k val]]
                                            (str (fmt-result k) " " (fmt-result val)))
                                          (sort-by (comp str key) v))) "}")
    (sequential? v) (str "[" (str/join " " (map fmt-result v)) "]")
    :else (str v)))

;; ═══════════════════════════════════════════════════════════════
;; Safe eval — catches errors, returns nil on failure
;; ═══════════════════════════════════════════════════════════════

(defn safe-eval
  "Eval an expression, return [result true] or [nil false] on error."
  [expr]
  (try
    (let [r (eval expr)]
      ;; Force lazy seqs and convert to vec for consistency
      (let [result (cond
                     (and (seq? r) (not (list? r))) (vec r)
                     (seq? r) (vec r)
                     :else r)]
        [result true]))
    (catch Exception _e
      [nil false])))

;; ═══════════════════════════════════════════════════════════════
;; Lambda expansion table — what named functions ARE as lambdas
;; ═══════════════════════════════════════════════════════════════

(def lambda-expansions
  "Map of function name → lambda notation string."
  {'inc       "(λx. (+ x 1))"
   'dec       "(λx. (- x 1))"
   'even?     "(λx. (= (mod x 2) 0))"
   'odd?      "(λx. (not= (mod x 2) 0))"
   'zero?     "(λx. (= x 0))"
   'pos?      "(λx. (> x 0))"
   'neg?      "(λx. (< x 0))"
   'identity  "(λx. x)"
   'not       "(λx. (not x))"
   'abs       "(λx. (if (neg? x) (- x) x))"
   'str       "(λx. (str x))"
   'count     "(λx. (count x))"
   'first     "(λx. (first x))"
   'last      "(λx. (last x))"
   'rest      "(λx. (rest x))"
   'reverse   "(λx. (reverse x))"
   'sort      "(λx. (sort x))"
   'distinct  "(λx. (distinct x))"
   'flatten   "(λx. (flatten x))"
   'empty?    "(λx. (empty? x))"
   'nil?      "(λx. (= x nil))"
   'some?     "(λx. (not= x nil))"
   'number?   "(λx. (number? x))"
   'string?   "(λx. (string? x))"
   'keyword?  "(λx. (keyword? x))"
   'vector?   "(λx. (vector? x))"
   'map?      "(λx. (map? x))"
   'set?      "(λx. (set? x))"
   'coll?     "(λx. (coll? x))"
   'true?     "(λx. (= x true))"
   'false?    "(λx. (= x false))"
   'keys      "(λx. (keys x))"
   'vals      "(λx. (vals x))"})

(defn lambda-expand
  "If sym has a lambda expansion, return it. Otherwise return (str sym)."
  [sym]
  (get lambda-expansions sym (str sym)))

;; ═══════════════════════════════════════════════════════════════
;; Notation formatters
;; ═══════════════════════════════════════════════════════════════

(defn fmt-sexpr
  "Format expression and result as s-expr notation."
  [expr result]
  (str (pr-str expr) " → " (fmt-result result)))

(defn fmt-raw-binary
  "Format a binary op as raw math: a + b = result"
  [op-sym a b result]
  (let [sym (case op-sym
              + "+" - "-" * "*" / "/" mod "mod" rem "rem" quot "quot"
              < "<" > ">" <= "<=" >= ">=" = "=" not= "!="
              bit-and "bit-and" bit-or "bit-or" bit-xor "bit-xor"
              bit-shift-left "bit-shift-left" bit-shift-right "bit-shift-right"
              (str op-sym))]
    (str a " " sym " " b " = " (fmt-result result))))

(defn fmt-raw-unary
  "Format a unary op as raw math: op(a) = result"
  [op-sym a result]
  (str (name op-sym) "(" a ") = " (fmt-result result)))

(defn fmt-raw-compound
  "Format compound expressions in raw math notation."
  [text result]
  (str text " = " (fmt-result result)))

(defn fmt-lambda-binary
  "Format binary op as lambda: (λx. λy. (op x y) a b) → result"
  [op-sym a b result]
  (str "(λx. λy. (" op-sym " x y) " a " " b ") → " (fmt-result result)))

(defn fmt-lambda-unary
  "Format unary op as lambda: (λx. (op x) a) → result"
  [op-sym a result]
  (str "(λx. (" op-sym " x) " a ") → " (fmt-result result)))

(defn fmt-lambda-hof
  "Format higher-order function call with lambda-expanded fn arg.
   (map inc [1 2 3]) → (map (λx. (+ x 1)) [1 2 3]) → [2 3 4]"
  [hof f-sym args result]
  (let [f-lambda (lambda-expand f-sym)
        args-str (str/join " " (map pr-str args))]
    (str "(" hof " " f-lambda " " args-str ") → " (fmt-result result))))

;; ═══════════════════════════════════════════════════════════════
;; Math expression generators
;; ═══════════════════════════════════════════════════════════════

;; ── Tier 1: Single operation ─────────────────────────────────

(def binary-arith-ops '[+ - *])
(def comparison-ops  '[< > <= >= = not=])
(def unary-ops       '[inc dec])
(def predicate-ops   '[zero? pos? neg? even? odd?])
(def bitwise-ops     '[bit-and bit-or bit-xor])

(defn gen-addition []
  (let [a (rand-digits) b (rand-digits)
        expr (list '+ a b)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:raw :sexpr :lambda])
        :raw    (fmt-raw-binary '+ a b result)
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-binary '+ a b result)))))

(defn gen-subtraction []
  (let [a (rand-digits) b (rand-digits)
        expr (list '- a b)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:raw :sexpr :lambda])
        :raw    (fmt-raw-binary '- a b result)
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-binary '- a b result)))))

(defn gen-multiplication []
  (let [a (rand-digits) b (rand-digits)
        expr (list '* a b)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:raw :sexpr :lambda])
        :raw    (fmt-raw-binary '* a b result)
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-binary '* a b result)))))

(defn gen-division []
  ;; Generate clean division: pick result and divisor, multiply for dividend
  (let [b (rand-pos 99)
        result (rand-digits)
        a (* result b)
        expr (list '/ a b)
        [r ok?] (safe-eval expr)]
    (when (and ok? (= r result))
      (case (rand-choice [:raw :sexpr :lambda])
        :raw    (fmt-raw-binary '/ a b r)
        :sexpr  (fmt-sexpr expr r)
        :lambda (fmt-lambda-binary '/ a b r)))))

(defn gen-mod []
  (let [a (rand-digits) b (rand-pos 99)
        expr (list 'mod a b)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:raw :sexpr :lambda])
        :raw    (fmt-raw-binary 'mod a b result)
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-binary 'mod a b result)))))

(defn gen-comparison []
  (let [op (rand-choice comparison-ops)
        a (rand-digits) b (rand-digits)
        expr (list op a b)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:raw :sexpr :lambda])
        :raw    (fmt-raw-binary op a b result)
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-binary op a b result)))))

(defn gen-unary []
  (let [op (rand-choice unary-ops)
        a (rand-digits)
        expr (list op a)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:raw :sexpr :lambda])
        :raw    (fmt-raw-unary op a result)
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-unary op a result)))))

(defn gen-predicate []
  (let [op (rand-choice predicate-ops)
        a (case op
            zero? (if (rand-bool) 0 (rand-digits))
            neg?  (if (rand-bool) (- (rand-pos 99)) (rand-digits))
            pos?  (rand-signed)
            (rand-digits))
        expr (list op a)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:raw :sexpr :lambda])
        :raw    (fmt-raw-unary op a result)
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-unary op a result)))))

(defn gen-boolean []
  (let [variant (rand-choice [:and :or :not])]
    (case variant
      :not (let [a (rand-bool)
                 expr (list 'not a)
                 [result ok?] (safe-eval expr)]
             (when ok?
               (case (rand-choice [:raw :sexpr :lambda])
                 :raw    (str "not " a " = " (fmt-result result))
                 :sexpr  (fmt-sexpr expr result)
                 :lambda (str "(λx. (not x) " a ") → " (fmt-result result)))))
      :and (let [a (rand-bool) b (rand-bool)
                 result (and a b)]
             (case (rand-choice [:raw :sexpr :lambda])
               :raw    (str a " and " b " = " (fmt-result result))
               :sexpr  (str "(and " a " " b ") → " (fmt-result result))
               :lambda (str "(λx. λy. (and x y) " a " " b ") → " (fmt-result result))))
      :or  (let [a (rand-bool) b (rand-bool)
                 result (or a b)]
             (case (rand-choice [:raw :sexpr :lambda])
               :raw    (str a " or " b " = " (fmt-result result))
               :sexpr  (str "(or " a " " b ") → " (fmt-result result))
               :lambda (str "(λx. λy. (or x y) " a " " b ") → " (fmt-result result)))))))

(defn gen-bitwise []
  (let [op (rand-choice bitwise-ops)
        a (rand-int* 0 255) b (rand-int* 0 255)
        expr (list op a b)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:raw :sexpr :lambda])
        :raw    (fmt-raw-binary op a b result)
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-binary op a b result)))))

(defn gen-bit-shift []
  (let [op (rand-choice '[bit-shift-left bit-shift-right])
        a (rand-int* 1 999) b (rand-int* 0 8)
        expr (list op a b)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:raw :sexpr :lambda])
        :raw    (fmt-raw-binary op a b result)
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-binary op a b result)))))

(defn gen-abs []
  (let [a (rand-signed)
        expr (list 'abs a)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:raw :sexpr :lambda])
        :raw    (fmt-raw-unary 'abs a result)
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-unary 'abs a result)))))

(defn gen-max-min []
  (let [op (rand-choice '[max min])
        a (rand-digits) b (rand-digits)
        expr (list op a b)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:raw :sexpr :lambda])
        :raw    (fmt-raw-binary op a b result)
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-binary op a b result)))))

;; ── Tier 2: Compound (2 operations) ─────────────────────────

(defn gen-compound-arith []
  (let [variant (rand-choice [:add-mul :sub-mul :mul-add :nested-pred
                               :max-expr :min-expr :square :double])]
    (case variant
      :add-mul
      (let [a (rand-digits) b (rand-digits) c (rand-int* 1 20)
            expr (list '* (list '+ a b) c)
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:raw :sexpr :lambda])
            :raw    (fmt-raw-compound (str "(" a " + " b ") * " c) result)
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "(λa. λb. λc. (* (+ a b) c) " a " " b " " c ") → " (fmt-result result)))))

      :sub-mul
      (let [a (rand-digits) b (rand-digits) c (rand-int* 1 20)
            expr (list '* (list '- a b) c)
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:raw :sexpr :lambda])
            :raw    (fmt-raw-compound (str "(" a " - " b ") * " c) result)
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "(λa. λb. λc. (* (- a b) c) " a " " b " " c ") → " (fmt-result result)))))

      :mul-add
      (let [a (rand-int* 0 9) b (rand-int* 0 9) c (rand-int* 0 9) d (rand-int* 0 9)
            expr (list '+ (list '* a b) (list '* c d))
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:raw :sexpr :lambda])
            :raw    (fmt-raw-compound (str a " * " b " + " c " * " d) result)
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "(λa. λb. λc. λd. (+ (* a b) (* c d)) " a " " b " " c " " d ") → " (fmt-result result)))))

      :nested-pred
      (let [pred (rand-choice '[even? odd? zero? pos? neg?])
            op (rand-choice '[+ - *])
            a (rand-digits) b (rand-digits)
            expr (list pred (list op a b))
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:raw :sexpr :lambda])
            :raw    (str (name pred) "(" a " " (name op) " " b ") = " (fmt-result result))
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "(λa. λb. (" pred " (" op " a b)) " a " " b ") → " (fmt-result result)))))

      :max-expr
      (let [a (rand-digits) b (rand-digits) c (rand-digits) d (rand-digits)
            expr (list 'max (list '+ a b) (list '- c d))
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:raw :sexpr :lambda])
            :raw    (fmt-raw-compound (str "max(" a " + " b ", " c " - " d ")") result)
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "(λa. λb. λc. λd. (max (+ a b) (- c d)) " a " " b " " c " " d ") → " (fmt-result result)))))

      :min-expr
      (let [a (rand-digits) b (rand-digits) c (rand-digits) d (rand-digits)
            expr (list 'min (list '* a b) (list '+ c d))
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:raw :sexpr :lambda])
            :raw    (fmt-raw-compound (str "min(" a " * " b ", " c " + " d ")") result)
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "(λa. λb. λc. λd. (min (* a b) (+ c d)) " a " " b " " c " " d ") → " (fmt-result result)))))

      :square
      (let [x (rand-int* 0 99)
            expr (list '* x x)
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:raw :sexpr :lambda])
            :raw    (str x "² = " (fmt-result result))
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "(λx. (* x x) " x ") → " (fmt-result result)))))

      :double
      (let [x (rand-digits)
            expr (list '+ x x)
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:raw :sexpr :lambda])
            :raw    (str "2 * " x " = " (fmt-result result))
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "(λx. (+ x x) " x ") → " (fmt-result result))))))))

;; ── Tier 3: Nested (3 operations) ───────────────────────────

(defn gen-nested-arith []
  (let [variant (rand-choice [:full-nest :chain :compare-compound])]
    (case variant
      :full-nest
      (let [a (rand-int* 0 50) b (rand-int* 0 50) c (rand-int* 0 50)
            d (rand-int* 0 50) e (rand-int* 0 50)
            expr (list '+ (list '* (list '+ a b) (list '- c d)) e)
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:raw :sexpr :lambda])
            :raw    (fmt-raw-compound (str "((" a " + " b ") * (" c " - " d ")) + " e) result)
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "(λa. λb. λc. λd. λe. (+ (* (+ a b) (- c d)) e) "
                         a " " b " " c " " d " " e ") → " (fmt-result result)))))

      :chain
      (let [a (rand-digits) b (rand-digits) c (rand-int* 0 50)
            expr (list '+ (list 'abs (list '- a b)) c)
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:raw :sexpr :lambda])
            :raw    (fmt-raw-compound (str "abs(" a " - " b ") + " c) result)
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "(λa. λb. λc. (+ (abs (- a b)) c) " a " " b " " c ") → " (fmt-result result)))))

      :compare-compound
      (let [cmp (rand-choice '[< > <= >= =])
            a (rand-digits) b (rand-digits) c (rand-int* 0 9) d (rand-int* 0 9)
            expr (list cmp (list '+ a b) (list '* c d))
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:raw :sexpr :lambda])
            :raw    (fmt-raw-compound (str "(" a " + " b ") " (name cmp) " (" c " * " d ")") result)
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "(λa. λb. λc. λd. (" cmp " (+ a b) (* c d)) "
                         a " " b " " c " " d ") → " (fmt-result result))))))))


;; ═══════════════════════════════════════════════════════════════
;; Clojure.core generators — eval'd in babashka
;; ═══════════════════════════════════════════════════════════════

;; Helper: build sexpr or lambda notation for HOF calls
(defn gen-hof-example
  "Generate a higher-order function example.
   hof-sym: 'map, 'filter, etc.
   f-sym: 'inc, 'even?, etc.
   args: remaining args after f
   Returns formatted string or nil."
  [hof-sym f-sym args]
  (let [expr (apply list hof-sym f-sym args)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-hof hof-sym f-sym args result)))))

;; ── Sequence operations ──────────────────────────────────────

(defn gen-map []
  (let [f (rand-choice '[inc dec])
        xs (rand-int-list 2 8)]
    (gen-hof-example 'map f [xs])))

(defn gen-map-math []
  ;; map with inline math fn — only sexpr since lambda is complex
  (let [op (rand-choice '[+ - *])
        n (rand-int* 1 10)
        xs (rand-int-list 2 6)
        expr (list 'map (list 'fn ['x] (list op 'x n)) xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (let [notation (rand-choice [:sexpr :lambda])]
        (case notation
          :sexpr  (fmt-sexpr expr result)
          :lambda (str "(map (λx. (" op " x " n ")) " (pr-str xs) ") → " (fmt-result result)))))))

(defn gen-filter []
  (let [pred (rand-choice '[even? odd? pos? neg? zero?])
        xs (rand-signed-list 4 10)]
    (gen-hof-example 'filter pred [xs])))

(defn gen-remove []
  (let [pred (rand-choice '[even? odd? nil? zero?])
        xs (if (= pred 'nil?)
             (vec (map #(if (< (.nextDouble *rng*) 0.3) nil %) (rand-int-list 4 8)))
             (rand-signed-list 4 8))]
    (gen-hof-example 'remove pred [xs])))

(defn gen-reduce-add []
  (let [xs (rand-int-list 2 8)
        expr (list 'reduce '+ xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (str "(reduce (λacc. λx. (+ acc x)) " (pr-str xs) ") → " (fmt-result result))))))

(defn gen-reduce-mul []
  (let [xs (rand-small-list 2 5)
        expr (list 'reduce '* xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (str "(reduce (λacc. λx. (* acc x)) " (pr-str xs) ") → " (fmt-result result))))))

(defn gen-reduce-max-min []
  (let [op (rand-choice '[max min])
        xs (rand-int-list 2 8)
        expr (list 'reduce op xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (str "(reduce (λacc. λx. (" op " acc x)) " (pr-str xs) ") → " (fmt-result result))))))

(defn gen-apply []
  (let [op (rand-choice '[+ * max min])
        xs (if (= op '*) (rand-small-list 2 5) (rand-int-list 2 7))
        expr (list 'apply op xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-first-last-rest []
  (let [op (rand-choice '[first last rest])
        xs (rand-int-list 3 8)
        expr (list op xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-take-drop []
  (let [op (rand-choice '[take drop])
        xs (rand-int-list 4 10)
        n (rand-int* 1 (min 5 (count xs)))
        expr (list op n xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-take-drop-while []
  (let [op (rand-choice '[take-while drop-while])
        pred (rand-choice '[even? odd? pos?])
        xs (rand-signed-list 4 8)
        expr (list op pred xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-hof op pred [xs] result)))))

(defn gen-nth []
  (let [xs (rand-int-list 3 8)
        n (rand-int* 0 (dec (count xs)))
        expr (list 'nth xs n)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-count []
  (let [xs (rand-int-list 1 10)
        expr (list 'count xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-sort []
  (let [xs (rand-int-list 3 8)
        expr (list 'sort xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-reverse []
  (let [xs (rand-int-list 3 7)
        expr (list 'reverse xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-concat []
  (let [xs (rand-int-list 2 5) ys (rand-int-list 2 5)
        expr (list 'concat xs ys)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-flatten []
  (let [a (rand-int-list 1 3) b (rand-int-list 1 3)
        expr (list 'flatten [a b])
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-range []
  (let [variant (rand-choice [:n :from-to :step])
        [expr ok?-pre] (case variant
                         :n      [(list 'range (rand-int* 2 15)) true]
                         :from-to (let [a (rand-int* 0 10) b (+ a (rand-int* 2 10))]
                                    [(list 'range a b) true])
                         :step   (let [a 0 b (rand-int* 10 50) s (rand-int* 2 7)]
                                   [(list 'range a b s) true]))
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-repeat []
  (let [n (rand-int* 2 7) v (rand-digits)
        expr (list 'repeat n v)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-distinct []
  (let [xs (vec (concat (rand-int-list 3 5) (rand-int-list 2 3)))
        expr (list 'distinct xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-interleave []
  (let [xs (rand-int-list 2 4) ys (rand-int-list 2 4)
        expr (list 'interleave xs ys)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-partition []
  (let [n (rand-int* 2 4)
        xs (rand-int-list (* n 2) (* n 4))
        expr (list 'partition n xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-frequencies []
  (let [xs (vec (repeatedly (rand-int* 4 10) #(rand-int* 0 5)))
        expr (list 'frequencies xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-group-by []
  (let [pred (rand-choice '[even? odd?])
        xs (rand-int-list 4 8)
        expr (list 'group-by pred xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-hof 'group-by pred [xs] result)))))

(defn gen-zipmap []
  (let [ks (vec (take (rand-int* 2 5) [:a :b :c :d :e :f]))
        vs (rand-int-list (count ks) (count ks))
        expr (list 'zipmap ks vs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-mapcat []
  (let [xs (rand-int-list 3 5)
        ;; mapcat (fn [x] [x (* x 2)])
        expr (list 'mapcat (list 'fn ['x] ['x (list '* 'x 2)]) xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (let [notation (rand-choice [:sexpr :lambda])]
        (case notation
          :sexpr  (fmt-sexpr expr result)
          :lambda (str "(mapcat (λx. [x (* x 2)]) " (pr-str xs) ") → " (fmt-result result)))))))

(defn gen-some-every []
  (let [op (rand-choice '[some every?])
        pred (rand-choice '[even? odd? pos? neg? zero?])
        xs (rand-signed-list 3 7)
        expr (list op pred xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (fmt-lambda-hof op pred [xs] result)))))

(defn gen-keep []
  (let [;; keep with fn that returns val or nil
        xs (rand-int-list 4 8)
        ;; (keep #(when (even? %) (* % 2)) xs)
        expr (list 'keep (list 'fn ['x] (list 'when (list 'even? 'x) (list '* 'x 2))) xs)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

;; ── Collection operations ────────────────────────────────────

(defn gen-assoc []
  (let [k (rand-choice [:a :b :c :x :y :name :age :score])
        v (rand-digits)
        m {:a 1 :b 2}
        expr (list 'assoc m k v)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-dissoc []
  (let [m {:a 1 :b 2 :c 3}
        k (rand-choice [:a :b :c])
        expr (list 'dissoc m k)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-get []
  (let [m {:a 10 :b 20 :c 30}
        k (rand-choice [:a :b :c :d])
        expr (list 'get m k)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-get-in []
  (let [m {:a {:x 1 :y 2} :b {:x 3 :y 4}}
        ks (rand-choice [[:a :x] [:a :y] [:b :x] [:b :y] [:c :x]])
        expr (list 'get-in m ks)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-update []
  (let [k (rand-choice [:a :b :count :score])
        v (rand-digits)
        m {k v}
        f (rand-choice '[inc dec])
        expr (list 'update m k f)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (str "(update " (pr-str m) " " k " " (lambda-expand f) ") → " (fmt-result result))))))

(defn gen-merge []
  (let [m1 {:a 1 :b 2}
        k (rand-choice [:b :c :d])
        v (rand-digits)
        m2 {k v}
        expr (list 'merge m1 m2)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-select-keys []
  (let [m {:a 1 :b 2 :c 3 :d 4}
        ks (vec (take (rand-int* 1 3) (shuffle [:a :b :c :d])))
        expr (list 'select-keys m ks)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-keys-vals []
  (let [op (rand-choice '[keys vals])
        n (rand-int* 2 5)
        m (into {} (map (fn [i] [(keyword (str (char (+ 97 i)))) (rand-digits)])
                        (range n)))
        expr (list op m)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-conj []
  (let [xs (rand-int-list 2 5)
        v (rand-digits)
        expr (list 'conj xs v)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-into []
  (let [xs (rand-int-list 2 4) ys (rand-int-list 2 4)
        expr (list 'into xs ys)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-contains []
  (let [m {:a 1 :b 2 :c 3}
        k (rand-choice [:a :b :d :e])
        expr (list 'contains? m k)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-empty []
  (let [coll (rand-choice [[] {} #{} [1 2] {:a 1} #{1}])
        expr (list 'empty? coll)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

;; ── String operations ────────────────────────────────────────

(def sample-words ["hello" "world" "foo" "bar" "baz" "clojure"
                   "lambda" "verbum" "alpha" "beta" "gamma"])

(defn gen-str-concat []
  (let [a (rand-choice sample-words) b (rand-choice sample-words)
        expr (list 'str a b)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-str-count []
  (let [w (rand-choice sample-words)
        expr (list 'count w)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-subs []
  (let [w (rand-choice sample-words)
        start (rand-int* 0 (max 0 (- (count w) 2)))
        end (rand-int* (inc start) (count w))
        expr (list 'subs w start end)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-str-join []
  (let [ws (vec (take (rand-int* 2 5) (shuffle sample-words)))
        sep (rand-choice [" " ", " "-" "/"])
        expr (list 'clojure.string/join sep ws)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-str-upper-lower []
  (let [op (rand-choice '[clojure.string/upper-case clojure.string/lower-case])
        w (rand-choice (if (= op 'clojure.string/upper-case)
                         sample-words
                         ["Hello" "WORLD" "FooBar" "LAMBDA" "Clojure"]))
        expr (list op w)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-str-trim []
  (let [w (rand-choice sample-words)
        expr (list 'clojure.string/trim (str "  " w "  "))
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-str-includes []
  (let [w (rand-choice sample-words)
        sub (subs w 0 (min 3 (count w)))
        expr (list 'clojure.string/includes? w sub)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-str-starts-ends []
  (let [op (rand-choice '[clojure.string/starts-with? clojure.string/ends-with?])
        w (rand-choice sample-words)
        sub (if (= op 'clojure.string/starts-with?)
              (subs w 0 (min 2 (count w)))
              (subs w (max 0 (- (count w) 2))))
        expr (list op w sub)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-str-replace []
  (let [w (rand-choice ["hello world" "foo bar baz" "one two three"])
        [from to] (rand-choice [["o" "0"] ["a" "@"] ["e" "3"]])
        expr (list 'clojure.string/replace w from to)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

;; ── Type predicates ──────────────────────────────────────────

(defn gen-type-pred []
  (let [checks [['nil? nil true] ['nil? 42 false] ['nil? :foo false]
                 ['some? 42 true] ['some? nil false]
                 ['number? 42 true] ['number? "hi" false] ['number? :x false]
                 ['string? "hi" true] ['string? 42 false]
                 ['keyword? :foo true] ['keyword? "hi" false]
                 ['vector? [1 2] true] ['vector? {:a 1} false]
                 ['map? {:a 1} true] ['map? [1 2] false]
                 ['set? #{1 2} true] ['set? [1 2] false]
                 ['coll? [1 2] true] ['coll? {:a 1} true] ['coll? 42 false]
                 ['true? true true] ['true? false false] ['true? 1 false]
                 ['false? false true] ['false? true false] ['false? nil false]]
        [pred val result] (rand-choice checks)]
    (str "(" pred " " (pr-str val) ") → " (fmt-result result))))

;; ── Conditionals ─────────────────────────────────────────────

(defn gen-if []
  (let [variant (rand-choice [:bool :compare])]
    (case variant
      :bool (let [c (rand-bool) a (rand-digits) b (rand-digits)
                  expr (list 'if c a b)
                  [result ok?] (safe-eval expr)]
              (when ok? (fmt-sexpr expr result)))
      :compare (let [cmp (rand-choice '[< > =])
                     x (rand-digits) y (rand-digits)
                     a (rand-digits) b (rand-digits)
                     expr (list 'if (list cmp x y) a b)
                     [result ok?] (safe-eval expr)]
                 (when ok? (fmt-sexpr expr result))))))

(defn gen-when []
  (let [c (rand-bool) v (rand-digits)
        expr (list 'when c v)
        [result ok?] (safe-eval expr)]
    (when ok? (fmt-sexpr expr result))))

(defn gen-cond []
  (let [x (rand-signed)
        expr (list 'cond
                   (list 'neg? x) "negative"
                   (list 'zero? x) "zero"
                   :else "positive")
        [result ok?] (safe-eval expr)]
    (when ok? (fmt-sexpr expr result))))

;; ── Let bindings ─────────────────────────────────────────────

(defn gen-let []
  (let [variant (rand-choice [:add :mul :use-twice :nested])]
    (case variant
      :add (let [a (rand-digits) b (rand-digits)
                 expr (list 'let ['x a 'y b] (list '+ 'x 'y))
                 [result ok?] (safe-eval expr)]
             (when ok?
               (case (rand-choice [:sexpr :lambda])
                 :sexpr  (fmt-sexpr expr result)
                 :lambda (str "((λx. (λy. (+ x y)) " b ") " a ") → " (fmt-result result)))))
      :mul (let [a (rand-digits) b (rand-digits)
                 expr (list 'let ['x a 'y b] (list '* 'x 'y))
                 [result ok?] (safe-eval expr)]
             (when ok?
               (case (rand-choice [:sexpr :lambda])
                 :sexpr  (fmt-sexpr expr result)
                 :lambda (str "((λx. (λy. (* x y)) " b ") " a ") → " (fmt-result result)))))
      :use-twice
      (let [a (rand-digits)
            expr (list 'let ['x a] (list '+ 'x 'x))
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:sexpr :lambda])
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "((λx. (+ x x)) " a ") → " (fmt-result result)))))
      :nested
      (let [a (rand-digits) b (rand-digits)
            expr (list 'let ['x a 'y (list '+ 'x b)] (list '* 'y 2))
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:sexpr :lambda])
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "((λx. ((λy. (* y 2)) (+ x " b "))) " a ") → " (fmt-result result))))))))

;; ── Function definitions + application ───────────────────────

(defn gen-fn-apply []
  (let [variant (rand-choice [:defn :anon :higher-order])]
    (case variant
      :defn (let [op (rand-choice '[+ - *])
                  a (rand-digits) b (rand-digits)
                  result ({'+  (+ a b) '- (- a b) '* (* a b)} op)]
              (case (rand-choice [:sexpr :lambda])
                :sexpr  (str "(defn f [x y] (" op " x y)) (f " a " " b ") → " (fmt-result result))
                :lambda (str "(def f (λx. λy. (" op " x y))) (f " a " " b ") → " (fmt-result result))))
      :anon (let [a (rand-digits)
                  expr (list (list 'fn ['x] (list '+ (list '* 'x 'x) 1)) a)
                  [result ok?] (safe-eval expr)]
              (when ok?
                (case (rand-choice [:sexpr :lambda])
                  :sexpr  (fmt-sexpr expr result)
                  :lambda (str "((λx. (+ (* x x) 1)) " a ") → " (fmt-result result)))))
      :higher-order
      (let [a (rand-digits) b (rand-digits)
            expr (list (list 'fn ['f 'x 'y] (list 'f 'x 'y)) '+ a b)
            [result ok?] (safe-eval expr)]
        (when ok?
          (case (rand-choice [:sexpr :lambda])
            :sexpr  (fmt-sexpr expr result)
            :lambda (str "((λf. λx. λy. (f x y)) + " a " " b ") → " (fmt-result result))))))))

;; ── Higher-order: comp, partial, juxt, identity ──────────────

(defn gen-comp []
  (let [a (rand-digits)
        ;; (comp inc #(* % 2)) → inc(a*2) = a*2+1
        expr (list (list 'comp 'inc (list 'fn ['x] (list '* 'x 2))) a)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (str "((λx. (+ (* x 2) 1)) " a ") → " (fmt-result result))))))

(defn gen-partial []
  (let [op (rand-choice '[+ * -])
        a (rand-digits) b (rand-digits)
        expr (list (list 'partial op a) b)
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (str "((λx. (" op " " a " x)) " b ") → " (fmt-result result))))))

(defn gen-juxt []
  (let [x (rand-digits)
        expr (list (list 'juxt 'inc 'dec) x)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-identity-constantly []
  (let [op (rand-choice [:identity :constantly])]
    (case op
      :identity (let [v (rand-digits)
                      expr (list 'identity v)
                      [result ok?] (safe-eval expr)]
                  (when ok? (fmt-sexpr expr result)))
      :constantly (let [v (rand-digits) x (rand-digits)
                        expr (list (list 'constantly v) x)
                        [result ok?] (safe-eval expr)]
                    (when ok? (fmt-sexpr expr result))))))

;; ── Compound clojure (2+ operations composed) ───────────────

(defn gen-filter-map []
  (let [xs (rand-int-list 4 8)
        expr (list 'filter 'even? (list 'map 'inc xs))
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (str "(filter (λx. (= (mod x 2) 0)) (map (λx. (+ x 1)) " (pr-str xs) ")) → " (fmt-result result))))))

(defn gen-map-filter []
  (let [xs (rand-int-list 4 8)
        expr (list 'map (list 'fn ['x] (list '* 'x 'x)) (list 'filter 'even? xs))
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (str "(map (λx. (* x x)) (filter (λx. (= (mod x 2) 0)) " (pr-str xs) ")) → " (fmt-result result))))))

(defn gen-reduce-map []
  (let [xs (rand-int-list 3 6)
        expr (list 'reduce '+ (list 'map (list 'fn ['x] (list '* 'x 'x)) xs))
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (str "(reduce (λacc. λx. (+ acc x)) (map (λx. (* x x)) " (pr-str xs) ")) → " (fmt-result result))))))

(defn gen-count-filter []
  (let [xs (rand-int-list 5 10)
        pred (rand-choice '[even? odd?])
        expr (list 'count (list 'filter pred xs))
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-first-filter []
  (let [xs (rand-int-list 5 10)
        pred (rand-choice '[even? odd?])
        expr (list 'first (list 'filter pred xs))
        [result ok?] (safe-eval expr)]
    (when (and ok? (some? result))
      (fmt-sexpr expr result))))

(defn gen-last-sort []
  (let [xs (rand-int-list 3 7)
        expr (list 'last (list 'sort xs))
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-take-sort []
  (let [xs (rand-int-list 5 10)
        n (rand-int* 2 4)
        expr (list 'take n (list 'sort xs))
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

(defn gen-sum-range []
  (let [n (rand-int* 2 15)
        expr (list 'reduce '+ (list 'range n))
        [result ok?] (safe-eval expr)]
    (when ok?
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        :lambda (str "(reduce (λacc. λx. (+ acc x)) (range " n ")) → " (fmt-result result))))))

;; ── Set operations ───────────────────────────────────────────

(defn gen-set-ops []
  (let [op (rand-choice ['clojure.set/union 'clojure.set/intersection
                          'clojure.set/difference])
        a (set (take (rand-int* 2 5) (shuffle (range 1 10))))
        b (set (take (rand-int* 2 5) (shuffle (range 1 10))))
        expr (list op a b)
        [result ok?] (safe-eval expr)]
    (when ok?
      (fmt-sexpr expr result))))

;; ── Threading macros ─────────────────────────────────────────

(defn gen-threading []
  (let [xs (rand-int-list 4 8)
        n (rand-int* 2 4)
        ;; ->> threading: (->> xs (map inc) (filter even?) (take n))
        expr (list '->> xs (list 'map 'inc) (list 'filter 'even?) (list 'take n))
        [result ok?] (safe-eval expr)]
    (when ok?
      ;; Show both threaded and unthreaded
      (case (rand-choice [:sexpr :lambda])
        :sexpr  (fmt-sexpr expr result)
        ;; For lambda, show the expanded form
        :lambda (let [expanded (list 'take n (list 'filter 'even? (list 'map 'inc xs)))]
                  (fmt-sexpr expanded result))))))

;; ═══════════════════════════════════════════════════════════════
;; Master generator — weighted random selection
;; ═══════════════════════════════════════════════════════════════

(def generator-pool
  "Weighted pool: [generator-fn weight]"
  [;; Math — Tier 1
   [gen-addition 20]
   [gen-subtraction 15]
   [gen-multiplication 15]
   [gen-division 10]
   [gen-mod 8]
   [gen-comparison 15]
   [gen-unary 10]
   [gen-predicate 12]
   [gen-boolean 10]
   [gen-bitwise 8]
   [gen-bit-shift 6]
   [gen-abs 6]
   [gen-max-min 8]
   ;; Math — Tier 2
   [gen-compound-arith 20]
   ;; Math — Tier 3
   [gen-nested-arith 12]
   ;; Clojure — Sequences
   [gen-map 15]
   [gen-map-math 12]
   [gen-filter 15]
   [gen-remove 8]
   [gen-reduce-add 12]
   [gen-reduce-mul 8]
   [gen-reduce-max-min 8]
   [gen-apply 8]
   [gen-first-last-rest 10]
   [gen-take-drop 10]
   [gen-take-drop-while 6]
   [gen-nth 6]
   [gen-count 6]
   [gen-sort 8]
   [gen-reverse 6]
   [gen-concat 6]
   [gen-flatten 4]
   [gen-range 8]
   [gen-repeat 4]
   [gen-distinct 4]
   [gen-interleave 4]
   [gen-partition 4]
   [gen-frequencies 5]
   [gen-group-by 5]
   [gen-zipmap 4]
   [gen-mapcat 5]
   [gen-some-every 6]
   [gen-keep 4]
   ;; Clojure — Collections
   [gen-assoc 6]
   [gen-dissoc 4]
   [gen-get 6]
   [gen-get-in 4]
   [gen-update 6]
   [gen-merge 5]
   [gen-select-keys 4]
   [gen-keys-vals 5]
   [gen-conj 5]
   [gen-into 5]
   [gen-contains 4]
   [gen-empty 4]
   ;; Clojure — Strings
   [gen-str-concat 5]
   [gen-str-count 4]
   [gen-subs 5]
   [gen-str-join 5]
   [gen-str-upper-lower 4]
   [gen-str-trim 3]
   [gen-str-includes 4]
   [gen-str-starts-ends 4]
   [gen-str-replace 3]
   ;; Clojure — Type predicates
   [gen-type-pred 8]
   ;; Clojure — Conditionals
   [gen-if 8]
   [gen-when 5]
   [gen-cond 5]
   ;; Clojure — Let bindings
   [gen-let 10]
   ;; Clojure — Function def + apply
   [gen-fn-apply 10]
   ;; Clojure — Higher-order
   [gen-comp 6]
   [gen-partial 6]
   [gen-juxt 4]
   [gen-identity-constantly 3]
   ;; Clojure — Compound (2+ ops)
   [gen-filter-map 8]
   [gen-map-filter 8]
   [gen-reduce-map 8]
   [gen-count-filter 5]
   [gen-first-filter 5]
   [gen-last-sort 5]
   [gen-take-sort 5]
   [gen-sum-range 6]
   ;; Clojure — Sets
   [gen-set-ops 5]
   ;; Clojure — Threading
   [gen-threading 6]])

(defn- build-weighted-pool
  "Build flat vector for weighted random selection."
  [pool]
  (vec (mapcat (fn [[gen-fn weight]]
                 (repeat weight gen-fn))
               pool)))

(def ^:private flat-pool (build-weighted-pool generator-pool))

(defn generate-one
  "Generate a single training example. Returns string or nil."
  []
  (let [gen-fn (rand-choice flat-pool)]
    (gen-fn)))

(defn generate-examples
  "Generate n training examples. Returns vector of strings."
  [n seed]
  (binding [*rng* (java.util.Random. seed)]
    (loop [examples []
           attempts 0]
      (if (or (>= (count examples) n) (>= attempts (* n 3)))
        examples
        (let [ex (generate-one)]
          (recur (if ex (conj examples ex) examples)
                 (inc attempts)))))))

;; ═══════════════════════════════════════════════════════════════
;; Stats
;; ═══════════════════════════════════════════════════════════════

(defn example-stats
  "Compute stats from generated examples."
  [examples]
  (let [total (count examples)
        by-arrow (group-by #(cond
                              (str/includes? % " → ") :sexpr-or-lambda
                              (str/includes? % " = ") :raw
                              :else :other)
                           examples)
        lambda-count (count (filter #(str/includes? % "λ") examples))
        sexpr-count (- (count (:sexpr-or-lambda by-arrow)) lambda-count)
        raw-count (count (:raw by-arrow))]
    {:total total
     :raw raw-count
     :sexpr sexpr-count
     :lambda lambda-count
     :avg-length (when (pos? total)
                   (double (/ (reduce + (map count examples)) total)))}))

;; ═══════════════════════════════════════════════════════════════
;; CLI entry point
;; ═══════════════════════════════════════════════════════════════

(defn run
  "Generate BIOS flash training data.
   Prints one example per line to stdout. Stats to stderr."
  [{:keys [count seed] :or {count 2560000 seed 42}}]
  (let [_ (binding [*out* *err*]
            (println "BIOS Flash — Babashka Training Data Generator")
            (println (str "  Generating " count " examples (seed=" seed ")...")))
        t0 (System/currentTimeMillis)
        examples (generate-examples count seed)
        elapsed (/ (- (System/currentTimeMillis) t0) 1000.0)
        stats (example-stats examples)]

    ;; Output examples to stdout (one per line)
    (doseq [ex examples]
      (println ex))

    ;; Stats to stderr
    (binding [*out* *err*]
      (println)
      (println (str "  Generated: " (:total stats) " examples in " (format "%.1f" elapsed) "s"))
      (println (str "  Raw:       " (:raw stats)))
      (println (str "  S-expr:    " (:sexpr stats)))
      (println (str "  Lambda:    " (:lambda stats)))
      (println (str "  Avg chars: " (when (:avg-length stats) (format "%.1f" (:avg-length stats))))))))
