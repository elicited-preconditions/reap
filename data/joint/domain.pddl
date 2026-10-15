; REAP joint pick-and-place domain over the open tabletop (tabletop-manipulation), the confined shelf
; (confined-shelf) and the multi-level shelf (shelf-access). The three source domains have IDENTICAL
; pick/put schemas; their predicates are a union; the only mismatch is the object type (cylinder vs
; movable), resolved by cylinder - movable. Each problem names its source domain in its first line,
; "; reap-env: <domain>", which selects the simulator and feasibility checker for that problem.
(define (domain reap-unified)
  (:requirements :strips :typing :negative-preconditions)
  (:types
    movable cell color direction - object
    cylinder - movable
  )
  (:constants
    north east - direction
  )
  (:predicates
    (adjacent ?dir - direction ?cel1 - cell ?cel2 - cell)
    (occupied ?cel - cell ?obj - movable)
    (empty ?cel - cell)
    (holding ?obj - movable)
    (gripper-empty)
    (discarded ?obj - movable)
    (color-of ?obj - movable ?c - color)
  )
  (:action pick
    :parameters (?obj - movable ?cel - cell)
    :precondition (and
      (gripper-empty)
      (occupied ?cel ?obj)
    )
    :effect (and
      (not (gripper-empty))
      (not (occupied ?cel ?obj))
      (empty ?cel)
      (holding ?obj)
    )
  )
  (:action put
    :parameters (?obj - movable ?cel - cell)
    :precondition (and
      (holding ?obj)
      (empty ?cel)
    )
    :effect (and
      (gripper-empty)
      (occupied ?cel ?obj)
      (not (empty ?cel))
      (not (holding ?obj))
    )
  )
)
