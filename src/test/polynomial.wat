(module
  (func (export "polynomial") (param $x f64) (param $y f64) (result f64)
    (local $x2 f64)
    (local $y2 f64)
    (local $f f64)
    (local.set $x2
      (f64.mul
        (local.get $x)
        (local.get $x)))
    (local.set $y2
      (f64.mul
        (local.get $y)
        (local.get $y)))
    (local.set $f
      (f64.mul
        (f64.const 2)
        (f64.mul
          (local.get $x)
          (local.get $x2))))
    (local.set $f
      (f64.add
        (local.get $f)
        (f64.mul
          (f64.const 4)
          (f64.mul
            (local.get $x2)
            (local.get $y)))))
    (local.set $f
      (f64.add
        (local.get $f)
        (f64.mul
          (local.get $x)
          (f64.mul
            (local.get $y)
            (f64.mul
              (local.get $y2)
              (local.get $y2))))))
    (local.set $f
      (f64.add
        (local.get $f)
        (local.get $y2)))
    (f64.sub
      (local.get $f)
      (f64.const 7))))
