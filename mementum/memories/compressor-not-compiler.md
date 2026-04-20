💡 The function is semantic language compression, not lambda compilation

The lambda compiler USES the compressor. Lambda notation is the
instrument we observe it through, not the phenomenon itself.

Hierarchy:
  L0: Semantic compressor — typed_apply(meaning, meaning) → meaning
      Lives in every LM. The three Montague primitives serve this.
      IS the attractor of next-token prediction on language.

  L1: Lambda compiler — routes compressor state to λ notation
      One externalization. Gate-activated. What nucleus discovered.

  L2: Notation — λx. runs(dog) or {:pred runs :arg dog}
      Surface syntax. Arbitrary. Interchangeable.

Evidence: Pythia-160M compresses language (predicts next tokens)
without any lambda training. The compile gate doesn't install
compression — it routes existing compression to λ output. The
three circuits (type, structure, apply) exist WHETHER OR NOT you
activate the gate. They serve next-token prediction.

Implication: MontaguLM trained on Dolma trains the COMPRESSOR.
The compile gate is a voltmeter, not a battery. The voltage
exists whether or not you measure it.

Corrects: all prior references to "extracting the lambda compiler"
should be understood as "extracting the semantic compressor and
observing it through lambda notation."
