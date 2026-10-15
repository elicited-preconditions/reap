(define (domain tabletop-manipulation)
  (:requirements :strips :typing :negative-preconditions :derived-predicates :disjunctive-preconditions :existential-preconditions)
  
  (:types
    cell cylinder direction
  )
  
  (:constants
    north east - direction
  )
  
  (:predicates
    (adjacent ?dir - direction ?cel1 - cell ?cel2 - cell)  ; cell cel2 is adjacent to cel1 in direction dir
    (occupied ?cel - cell ?cyl - cylinder)              ; cell contains cylinder
    (empty ?cel - cell)                     ; cell is empty
    (holding ?cyl - cylinder)               ; gripper is holding cylinder cyl
    (gripper-empty)                         ; gripper is not holding anything
    (discarded ?cyl - cylinder)             ; cylinder is discarded
  )
  
  (:action pick
    :parameters (?cyl - cylinder ?cel - cell)
    :precondition (and
      (gripper-empty)
      (occupied ?cel ?cyl)
    )
    :effect (and
      (not (gripper-empty))
      (not (occupied ?cel ?cyl))
      (empty ?cel)
      (holding ?cyl)
    )
  )

  (:action put
    :parameters (?cyl - cylinder ?cel - cell)
    :precondition (and
      (holding ?cyl)
      (empty ?cel)
    )
    :effect (and
      (gripper-empty)
      (occupied ?cel ?cyl)
      (not (empty ?cel))
      (not (holding ?cyl))
    )  
  )
)
