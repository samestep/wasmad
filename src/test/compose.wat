(module
  (func $gf (param $x f64) (result f64)
    (call $g (call $f (local.get $x))))
  (func $fg (param $x f64) (result f64)
    (call $f (call $g (local.get $x))))
  (func $f (param $x f64) (result f64)
    (f64.mul (local.get $x) (local.get $x)))
  (func $g (param $x f64) (result f64)
    (f64.add (local.get $x) (f64.const 1))))
