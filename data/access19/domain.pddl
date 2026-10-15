;; Access-19 (tabletop-access-v2): a two-action pick/put domain on a shelf of 5 x 7 cells per level.
;; Physical feasibility (reachability, blocking by other objects) is not modelled here; the feasibility checker
;; (tampanda access19_v2.checker, a grasp-library lookup) prunes actions. The derived predicate
;;   (blocking ?obj ?goal)  <=>  ?obj occupies a cell c with (same-column-front c ?goal)
;; is a hint for the GNN; it never appears in :init.

(define (domain tabletop-access-v2)
  (:requirements :strips :typing :negative-preconditions
                 :disjunctive-preconditions :existential-preconditions
                 :derived-predicates)

  (:types
    cell movable direction
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
    (same-column-front ?cel-front - cell ?cel-back - cell)
    (blocking ?obj - movable ?goal - cell)
  )

  (:derived (blocking ?obj - movable ?goal - cell)
    (exists (?c - cell)
      (and (occupied ?c ?obj)
           (same-column-front ?c ?goal))))

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
      (not (holding ?obj))
      (not (empty ?cel))
      (occupied ?cel ?obj)
    )
  )
)
