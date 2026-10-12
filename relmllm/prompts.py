from __future__ import annotations


def build_grounding_prompt(query: str, width: int, height: int) -> tuple[str, str]:
    system = "Return strict JSON only."
    user = (
        "Use both aligned RGB and TIR. Ground the query after explicitly "
        "assessing each modality and their relation.\n"
        f"Query: {query}\n"
        f"Images are aligned and each image has size {width}x{height}.\n"
        "The bbox must be [x1, y1, x2, y2] with each coordinate normalized "
        f"to the range 0..1000 for a single aligned image of size {width}x{height}. "
        "Do not use pair-concatenated coordinates.\n"
        "Use numeric bbox values. Keep every sub_answer at most 8 words. "
        "Return one closed JSON object only.\n"
        '{"steps":['
        '{"type":"rgb_evidence","modality":"RGB",'
        '"bbox":[x1,y1,x2,y2] or null,'
        '"reliability":"high|medium|low","sub_answer":"short"},'
        '{"type":"tir_evidence","modality":"TIR",'
        '"bbox":[x1,y1,x2,y2] or null,'
        '"reliability":"high|medium|low","sub_answer":"short"},'
        '{"type":"relation_check","modality":"RGB|TIR|Both|None",'
        '"relation":"consistent|complementary|conflict|uncertain",'
        '"sub_answer":"short"}],'
        '"bbox":[x1,y1,x2,y2],'
        '"rgb_reliability":"high|medium|low",'
        '"tir_reliability":"high|medium|low",'
        '"relation":"consistent|complementary|conflict|uncertain",'
        '"dominant_modality":"RGB|TIR|Both|None",'
        '"evidence":"short"}'
    )
    return system, user


def build_vqa_direct_prompt(question: str) -> tuple[str, str]:
    system = "Answer RGB-T visual yes/no questions exactly."
    user = (
        "You are given two aligned images of the same scene: first RGB/visible, "
        "second thermal infrared.\n"
        "Answer the question with exactly one word: Yes or No. Do not explain.\n"
        f"Question: {question}"
    )
    return system, user


def build_vqa_relation_prompt(question: str) -> tuple[str, str]:
    system = "Answer RGB-T visual yes/no questions exactly."
    user = (
        "You are given two aligned images of the same scene: first RGB/visible, "
        "second thermal infrared.\n"
        f"Question: {question}\n"
        "Assess RGB and TIR evidence, their reliability and cross-modal relation "
        "before answering. The reasoning must contain 2 to 4 complete, "
        "image-specific sentences (35 to 100 words). Return one closed JSON "
        "object only:\n"
        '{"steps":['
        '{"type":"rgb_evidence","modality":"RGB",'
        '"reliability":"high|medium|low","sub_answer":"short"},'
        '{"type":"tir_evidence","modality":"TIR",'
        '"reliability":"high|medium|low","sub_answer":"short"},'
        '{"type":"relation_check","modality":"RGB|TIR|Both|None",'
        '"relation":"consistent|complementary|conflict|uncertain",'
        '"sub_answer":"short"}],'
        '"reasoning":"image-specific cross-modal reasoning",'
        '"rgb_reliability":"high|medium|low",'
        '"tir_reliability":"high|medium|low",'
        '"relation":"consistent|complementary|conflict|uncertain",'
        '"dominant_modality":"RGB|TIR|Both|None",'
        '"answer":"Yes|No"}'
    )
    return system, user
