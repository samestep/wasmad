(module
  (type $pair (struct (field $fst f64) (field $snd f64)))
  (func (export "cons") (param $a f64) (param $b f64) (result (ref $pair))
    (struct.new $pair
      (local.get $a)
      (local.get $b)))
  (func (export "div") (param $ab (ref $pair)) (result f64)
    (f64.div
      (struct.get $pair $fst
        (local.get $ab))
      (struct.get $pair $snd
        (local.get $ab)))))
