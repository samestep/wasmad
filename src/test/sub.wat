(module
  (func (export "sub") (param $a f64) (param $b f64) (result f64)
    (f64.sub (local.get $a) (local.get $b))))
