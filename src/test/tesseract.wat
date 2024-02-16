(module
  (func $f (param f64) (result f64)
    (f64.mul
      (local.tee 0
        (f64.mul
          (local.get 0)
          (local.get 0)))
      (local.get 0))))
