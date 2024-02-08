(module
 (type $0 (struct (field i32)))
 (type $1 (func (param (ref $0)) (result (ref $0))))
 (func $foo (param $0 (ref $0)) (result (ref $0))
  (local.get $0)
 )
)
