; tabletop-relocate: REAP's open-tabletop domain extended for BT-h (Cieslar et
; al., CoRL 2024) with a relocate action whose target cell is chosen during
; refinement (the paper's appendix, Learning-Based Baselines).
;
; This is a separate file; REAP's data/tabletop/domain.pddl is not changed.
;
; It adds the :derived predicates `south-of` and `occluder` (an occluder is
; south of the target, in the same column or up to 2 columns to the side; see
; below) and makes two changes relative to REAP's own domain:
;   1. `pick` additionally requires that no cylinder currently occludes the
;      target (via `occluder`). Without this gate, a task planner would just
;      pick the target directly, and there would be nothing to backtrack over.
;   2. A new `relocate(cyl)` action moves a held cylinder "out of the way"
;      WITHOUT binding a target cell -- it asserts only `parked`, never
;      `occupied`. A parked cylinder therefore drops out of `occluder`'s
;      computation unconditionally, regardless of where the controller
;      actually places it in the real continuous world. That gap -- the
;      abstract layer can never detect that a real cell choice recreated a
;      problem for a later blocker or the target -- is the long-horizon
;      dependency the classifier has to learn to anticipate.
;      This mirrors Cieslar et al.'s own Trays domain, where tray
;      assignment is likewise not tracked at the task-plan level.
;
; No goal-check dummy action (unlike Trays) is needed: the goal is always
; `(holding cylinder_TARGET)`, a fluent `pick` sets exactly correctly
; regardless of where blockers ended up -- there is no possible mismatch
; between the abstract goal-check and reality, only during intermediate
; relocations.

(define (domain tabletop-relocate)
  (:requirements :strips :typing :negative-preconditions
                 :derived-predicates :disjunctive-preconditions :existential-preconditions)

  (:types
    cell cylinder direction
  )

  (:constants
    north east - direction
  )

  (:predicates
    (adjacent ?dir - direction ?cel1 - cell ?cel2 - cell)  ; cel2 is adjacent to cel1 in direction dir
    (occupied ?cel - cell ?cyl - cylinder)                 ; cell contains cylinder
    (empty ?cel - cell)                                    ; cell is empty
    (holding ?cyl - cylinder)                              ; gripper is holding cylinder cyl
    (gripper-empty)                                        ; gripper is not holding anything
    (parked ?cyl - cylinder)                               ; relocated out of the way; the cell is
                                                            ; not tracked symbolically
    (south-of ?south - cell ?north - cell)                 ; south is anywhere below north in same column
    (col-offset-1 ?c1 - cell ?c2 - cell)                    ; c1, c2 exactly 1 column apart, either direction
    (col-offset-2 ?c1 - cell ?c2 - cell)                    ; c1, c2 exactly 2 columns apart, either direction
    (occluder ?occ - cylinder ?tgt - cylinder)              ; occ occludes tgt from the south (within 2 columns)
  )

  (:action pick
    :parameters (?cyl - cylinder ?cel - cell)
    :precondition (and
      (gripper-empty)
      (occupied ?cel ?cyl)
      (not (exists (?occ - cylinder) (occluder ?occ ?cyl)))
    )
    :effect (and
      (not (gripper-empty))
      (not (occupied ?cel ?cyl))
      (empty ?cel)
      (holding ?cyl)
    )
  )

  (:action relocate
    :parameters (?cyl - cylinder)
    :precondition (and
      (holding ?cyl)
      (not (parked ?cyl))
    )
    :effect (and
      (gripper-empty)
      (not (holding ?cyl))
      (parked ?cyl)
    )
  )

  ; --- derived predicates ---

  (:derived (south-of ?south - cell ?north - cell)
    (or
      (adjacent north ?south ?north)
      (exists (?mid - cell)
        (and
          (adjacent north ?south ?mid)
          (south-of ?mid ?north)
        )
      )
    )
  )

  ; --- lateral tolerance: an occluder can be up to 2 columns to the side of
  ; the target, on the south side. Both col-offset-* are non-recursive. ---

  (:derived (col-offset-1 ?c1 - cell ?c2 - cell)
    (or
      (adjacent east ?c1 ?c2)
      (adjacent east ?c2 ?c1)
    )
  )

  (:derived (col-offset-2 ?c1 - cell ?c2 - cell)
    (exists (?m - cell)
      (or
        (and (adjacent east ?c1 ?m) (adjacent east ?m ?c2))
        (and (adjacent east ?c2 ?m) (adjacent east ?m ?c1))
      )
    )
  )

  (:derived (occluder ?occ - cylinder ?tgt - cylinder)
    (exists (?occ-cell - cell ?tgt-cell - cell ?mid-cell - cell)
      (and
        (occupied ?occ-cell ?occ)
        (occupied ?tgt-cell ?tgt)
        (or
          ; any distance south, same column
          (south-of ?occ-cell ?tgt-cell)
          ; any distance south, +-1 column from target
          (and (col-offset-1 ?occ-cell ?mid-cell) (south-of ?mid-cell ?tgt-cell))
          ; any distance south, +-2 columns from target
          (and (col-offset-2 ?occ-cell ?mid-cell) (south-of ?mid-cell ?tgt-cell))
        )
      )
    )
  )
)
