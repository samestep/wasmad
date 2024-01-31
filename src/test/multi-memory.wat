(module
  (func (export "store") (param $a f64) (param $b f64)
    (f64.store $one
      (i32.const 0)
      (local.get $a))
    (f64.store $two
      (i32.const 0)
      (local.get $b)))
  (func (export "div") (result f64)
    (f64.div
      (f64.load $one
        (i32.const 0))
      (f64.load $two
        (i32.const 0))))
  (memory $one 1 1)
  (memory $two 1 1))
