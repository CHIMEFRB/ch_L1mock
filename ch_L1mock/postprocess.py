"""Library of L1 mock postprocessing tasks.

Postprocessing tasks are classes that:
    1) Can be initialized with only keyword arguments, and
    2) Are callable with the signature `task_instance(dediserser, ibeam,
    trigger_set)`
    4) Return a list of Event objects (which may be empty).
    3) They *may* modify trigger_set in place.

Feel free to write your own and register them in the INDEX.

"""

INDEX = {
        }

