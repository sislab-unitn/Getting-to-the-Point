from typing import Any, Dict, List, Optional, Tuple
from utils import CATEGORIES, COLORS, SHAPES, SHEENS
from world import Entity, ImgEntity, ObjEntity, World

# Adjective order: https://dictionary.cambridge.org/grammar/british-grammar/adjectives-order
# use the shape as the noun
# matte/glossy are more like phisical properties
PROPERTIES_ORDER = ("size", "sheen", "color", "shape")
PROPERTIES_ORDER_IMG = "category"

TEMPLATES = {
    # "properties": "What is the {} of the object?",
    "properties": "{}What is the {} of the object? Choose from [{}].",
    # "position_absolute": "Where is the {}?",
    "position_absolute": "Where is the {}? Choose from [{}].",
    # "position_relative": "Where is the {} positioned with respect to the {}?",
    "position_relative": "Where is the {} positioned with respect to the {}? Choose from [{}].",
    "size_relative": "Is the {} {} than the {}",  # e.g. "Is the {red square} {smaller/larger} than the {blue circle}?"
    # "distance_relative": "What is the closest object to the {}?",
    "distance_relative": "What is the closest object to the {}? Choose from [{}].",
    # "distance_relative": "What is closest to the {}?",
    "size_relative": "What is the size of the {} with respect to the {}? Choose from [{}].",
    "count": "How many objects are there?",
    "count_closed": "How many objects are there? Choose from [{}].",
    # "count_obj": "How many {}s are there?",
    # "count_obj_closed": "How many {}s are there? Choose from [{}].",
    "count_obj": "How many {} are there?",
    "count_obj_closed": "How many {} are there? Choose from [{}].",
}

# property values
PROP_VALUES = {
    "shape": SHAPES,
    "color": COLORS,
    "sheen": [s for s in SHEENS if s != "none"],
    "area": [
        "top left",
        "top center",
        "top right",
        "center left",
        "center",
        "center right",
        "bottom left",
        "bottom center",
        "bottom right",
    ],
    "area_vert": ["left", "center", "right"],
    "area_hor": ["top", "center", "bottom"],
    "relative": [
        "above left",
        "directly above",
        "above right",
        "directly left",
        "directly right",
        "below left",
        "directly below",
        "below right",
    ],
    "size": ["larger", "same", "smaller"],
    "category": CATEGORIES,
    "count": [str(c) for c in range(10)],
    "count_from_two": [str(c) for c in range(2, 10)],
}

PROP_DESCR = {
    "shape": "",
    "color": "",
    "sheen": "Sheen is a measure of the reflected light from a material. ",
    "category": "",
}

# def properties_questions(e: Entity) -> List[Tuple[str, str, str]]:
# def properties_questions(e: Entity, properties: List[str] = ["shape", "color", "sheen"]) -> Dict[str, str]:
# def properties_questions(properties: List[str] = ["shape", "color", "sheen"]) -> Dict[str, str]:
#     questions = {}
#     template = TEMPLATES["properties"]
#
#     # if isinstance(e, ObjEntity):
#     #     properties = PROPERTIES
#     # else:
#     #     raise Exception(f"Entities of class '{e.__class__}' are not supported to generate property questions.")
#
#     for property in properties:
#         q = template.format(property)
#         # a = getattr(e, property)
#         questions[property] = q
#
#     return questions


def get_object_reference(
    e: Entity, properties: List[str] = ["shape", "color", "sheen"]
) -> str:
    values = []
    if isinstance(e, ObjEntity):
        values = [
            getattr(e, property)
            for property in PROPERTIES_ORDER
            if property in properties and getattr(e, property) != "none"
        ]
    elif isinstance(e, ImgEntity):
        assert len(properties) == 1 and properties[0] == "category"
        values = [getattr(e, properties[0])]
    return " ".join(values)


def get_val_list(val_order: Optional[List[int]], val_list_name: str) -> str:
    return get_val_list_from_list(val_order, PROP_VALUES[val_list_name])


def get_val_list_from_list(val_order: Optional[List[int]], val_list: List[Any]) -> str:
    if val_order is None:
        ordered_val_list = val_list
    else:
        assert len(val_order) == len(val_list)
        ordered_val_list = []
        for pos in val_order:
            ordered_val_list.append(val_list[pos])
    return ", ".join(ordered_val_list)


def properties_questions(
    properties: List[str] = ["shape", "color", "sheen"]
) -> Dict[str, str]:
    questions = {}
    template = TEMPLATES["properties"]

    for property in properties:
        # q = template.format(property, PROP_VALUES[property])
        # questions[property] = q
        questions[property] = prop_question(property)

    return questions


def prop_question(property: str, val_order: Optional[List[int]] = None) -> str:
    template = TEMPLATES["properties"]

    # if val_order is None:
    #     values = PROP_VALUES[property]
    # else:
    #     assert len(val_order) == len(PROP_VALUES[property])
    #     values = []
    #     for pos in val_order:
    #         values.append(PROP_VALUES[property][pos])
    #
    # q = template.format(PROP_DESCR[property], property, ", ".join(values))
    val_list_str = get_val_list(val_order, property)
    q = template.format(PROP_DESCR[property], property, val_list_str)

    return q


# def absolute_position_question(e: Entity, properties: List[str] = ["shape", "color", "sheen"], val_order: Optional[List[int]] = None, val_list_name: str = "area") -> str:
#     template = TEMPLATES["position_absolute"]
#
#     values = [
#        getattr(e, property)
#         for property in PROPERTIES_ORDER
#         if property in properties and getattr(e, property) != "none"
#     ]
#     reference = " ".join(values)
#
#     if val_order is None:
#         val_list = PROP_VALUES[val_list_name]
#     else:
#         assert len(val_order) == len(PROP_VALUES[val_list_name])
#         val_list = []
#         for pos in val_order:
#             val_list.append(PROP_VALUES[val_list_name][pos])
#     question = template.format(reference, ", ".join(val_list))
#
#     return question


def absolute_position_question(
    e: Entity,
    properties: List[str] = ["shape", "color", "sheen"],
    val_order: Optional[List[int]] = None,
    val_list_name: str = "area",
) -> str:
    template = TEMPLATES["position_absolute"]

    reference = get_object_reference(e, properties)

    val_list_str = get_val_list(val_order, val_list_name)
    question = template.format(reference, val_list_str)

    return question


def relative_position_question(
    e1: Entity,
    e2: Entity,
    properties: List[str] = ["shape", "color", "sheen"],
    val_order: Optional[List[int]] = None,
) -> str:
    template = TEMPLATES["position_relative"]

    # TODO only properties with different values?
    e1_ref = get_object_reference(e1, properties)
    e2_ref = get_object_reference(e2, properties)

    val_list_str = get_val_list(val_order, "relative")

    question = template.format(e1_ref, e2_ref, val_list_str)

    return question


def relative_distance_question(
    e1: Entity,
    ents: List[Entity],
    properties: List[str] = ["shape", "color", "sheen"],
    val_order: Optional[List[int]] = None,
) -> str:
    template = TEMPLATES["distance_relative"]

    e1_ref = get_object_reference(e1, properties)
    other_refs = []
    for e in ents:
        other_refs.append(get_object_reference(e, properties))

    # val_list_str = get_val_list(val_order, "relative")
    if val_order is None:
        val_list = other_refs
    else:
        assert len(val_order) == len(other_refs)
        val_list = []
        for pos in val_order:
            val_list.append(other_refs[pos])
    val_list_str = ", ".join(val_list)

    question = template.format(e1_ref, val_list_str)

    return question


def relative_size_question(
    e1: Entity,
    e2: Entity,
    properties: List[str] = ["shape", "color", "sheen"],
    val_order: Optional[List[int]] = None,
) -> str:
    template = TEMPLATES["size_relative"]

    e1_ref = get_object_reference(e1, properties)
    e2_ref = get_object_reference(e2, properties)

    val_list_str = get_val_list(val_order, "size")

    question = template.format(e1_ref, e2_ref, val_list_str)

    return question


def count_question(
    properties: List[str] = ["shape", "color", "sheen"],
    val_order: Optional[List[int]] = None,
    ref_entity: Optional[Entity] = None,
    val_list: Optional[List[Any]] = None,
) -> str:
    if val_order is None:
        question = TEMPLATES["count"]
        if ref_entity is not None:
            template = TEMPLATES["count_obj"]
            # + "s" for plural
            ref_entity_ref = get_object_reference(ref_entity, properties) + "s"
            # special case for plus
            if ref_entity_ref.endswith("pluss"):
                ref_entity_ref += "es"
            question = template.format(ref_entity_ref)
    else:
        # if not count_from_two:
        #     val_list_str = get_val_list(val_order, "count")
        # else:
        # val_list_str = get_val_list(val_order, "count_from_two")
        assert val_list is not None
        val_list_str = get_val_list_from_list(val_order, val_list)
        if ref_entity is not None:
            template = TEMPLATES["count_obj_closed"]
            # + "s" for plural
            ref_entity_ref = get_object_reference(ref_entity, properties) + "s"
            # special case for plus
            if ref_entity_ref.endswith("pluss"):
                ref_entity_ref += "es"
            question = template.format(ref_entity_ref, val_list_str)
        else:
            template = TEMPLATES["count_closed"]
            question = template.format(val_list_str)

    return question
