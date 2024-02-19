(module
  (type $i32s (array (mut i32)))
  (type $f64s (array (mut f64)))
  (type $i32ss (array (mut (ref null $i32s))))
  (type $f64ss (array (mut (ref null $f64s))))

  (func (export "i32sNew") (param $n i32) (result (ref $i32s))
    (array.new_default $i32s
      (local.get $n)))
  (func (export "f64sNew") (param $n i32) (result (ref $f64s))
    (array.new_default $f64s
      (local.get $n)))
  (func (export "i32ssNew") (param $n i32) (result (ref $i32ss))
    (array.new_default $i32ss
      (local.get $n)))
  (func (export "f64ssNew") (param $n i32) (result (ref $f64ss))
    (array.new_default $f64ss
      (local.get $n)))

  (func (export "i32sLen") (param $a (ref null $i32s)) (result i32)
    (array.len $i32s
      (local.get $a)))
  (func (export "f64sLen") (param $a (ref null $f64s)) (result i32)
    (array.len $f64s
      (local.get $a)))
  (func (export "i32ssLen") (param $a (ref $i32ss)) (result i32)
    (array.len $i32ss
      (local.get $a)))
  (func (export "f64ssLen") (param $a (ref $f64ss)) (result i32)
    (array.len $f64ss
      (local.get $a)))

  (func (export "i32sGet") (param $a (ref null $i32s)) (param $i i32) (result i32)
    (array.get $i32s
      (local.get $a)
      (local.get $i)))
  (func (export "f64sGet") (param $a (ref null $f64s)) (param $i i32) (result f64)
    (array.get $f64s
      (local.get $a)
      (local.get $i)))
  (func (export "i32ssGet") (param $a (ref $i32ss)) (param $i i32) (result (ref null $i32s))
    (array.get $i32ss
      (local.get $a)
      (local.get $i)))
  (func (export "f64ssGet") (param $a (ref $f64ss)) (param $i i32) (result (ref null $f64s))
    (array.get $f64ss
      (local.get $a)
      (local.get $i)))

  (func (export "i32sSet") (param $a (ref null $i32s)) (param $i i32) (param $x i32)
    (array.set $i32s
      (local.get $a)
      (local.get $i)
      (local.get $x)))
  (func (export "f64sSet") (param $a (ref null $f64s)) (param $i i32) (param $x f64)
    (array.set $f64s
      (local.get $a)
      (local.get $i)
      (local.get $x)))
  (func (export "i32ssSet") (param $a (ref $i32ss)) (param $i i32) (param $x (ref null $i32s))
    (array.set $i32ss
      (local.get $a)
      (local.get $i)
      (local.get $x)))
  (func (export "f64ssSet") (param $a (ref $f64ss)) (param $i i32) (param $x (ref null $f64s))
    (array.set $f64ss
      (local.get $a)
      (local.get $i)
      (local.get $x)))

  (func (export "f") (param $i i32) (param $j i32) (param $iss (ref $i32ss)) (param $jss (ref $i32ss)) (param $xss (ref $f64ss)) (param $yss (ref $f64ss)) (result f64)
    (local $k i32)
    (local $l i32)
    (array.set $i32ss
      (local.get $jss)
      (local.get $j)
      (array.get $i32ss
        (local.get $iss)
        (local.get $i)))
    (array.set $i32s
      (array.get $i32ss
        (local.get $jss)
        (local.get $j))
      (local.get $i)
      (array.get $i32s
        (array.get $i32ss
          (local.get $iss)
          (local.get $i))
        (local.get $j)))
    (array.set $i32s
      (array.get $i32ss
        (local.get $iss)
        (local.get $j))
      (local.get $i)
      (array.get $i32s
        (array.get $i32ss
          (local.get $jss)
          (local.get $i))
        (local.get $j)))
    (array.set $i32ss
      (local.get $iss)
      (local.get $j)
      (array.get $i32ss
        (local.get $jss)
        (local.get $i)))
    (local.set $k
      (array.get $i32s
        (array.get $i32ss
          (local.get $iss)
          (local.get $j))
        (local.get $i)))
    (local.set $l
      (array.get $i32s
        (array.get $i32ss
          (local.get $jss)
          (local.get $j))
        (local.get $i)))
    (array.set $f64ss
      (local.get $yss)
      (local.get $l)
      (array.get $f64ss
        (local.get $xss)
        (local.get $k)))
    (array.set $f64s
      (array.get $f64ss
        (local.get $yss)
        (local.get $l))
      (local.get $k)
      (array.get $f64s
        (array.get $f64ss
          (local.get $xss)
          (local.get $k))
        (local.get $l)))
    (array.set $f64s
      (array.get $f64ss
        (local.get $xss)
        (local.get $l))
      (local.get $k)
      (array.get $f64s
        (array.get $f64ss
          (local.get $yss)
          (local.get $k))
        (local.get $l)))
    (array.set $f64ss
      (local.get $xss)
      (local.get $l)
      (array.get $f64ss
        (local.get $yss)
        (local.get $k)))
    (f64.mul
      (array.get $f64s
        (array.get $f64ss
          (local.get $xss)
          (local.get $l))
        (local.get $k))
      (array.get $f64s
        (array.get $f64ss
          (local.get $yss)
          (local.get $l))
        (local.get $k)))))
