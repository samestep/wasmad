(module
  (func (export "div") (param $a f64) (param $b f64) (result f64)
    (f64.div
      (local.get $a)
      (local.get $b))))
