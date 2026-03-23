from qwen_vl_utils import process_vision_info


class GenerationCollator:
    def __init__(
        self,
        processor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):
        self.processor.tokenizer.padding_side = "left"  # type: ignore
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token  # type: ignore

        questions = []
        conversations = []
        answers = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            questions.append(question)
            answers.append(answer)
            sample_ids.append(sample_id)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": (
                                f"{self.instruction} {question}"
                                if self.instruction
                                else question
                            ),
                        },
                    ],
                }
            ]

            conversations.append(conversation)

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in conversations
        ]
        image_inputs, video_inputs = process_vision_info(conversations)  # type: ignore
        input_ids = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        questions = self.processor.tokenizer(  # type: ignore
            questions, return_tensors="pt", padding=True
        )

        return input_ids, questions, answers, sample_ids


class CoordinatesCollator:
    def __init__(
        self,
        processor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):
        self.processor.tokenizer.padding_side = "left"  # type: ignore
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token  # type: ignore

        questions = []
        conversations = []
        answers = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            questions.append(question)
            sample_ids.append(sample_id)

            assert (
                "Coordinates:" in answer
            ), "Answer must contain 'Coordinates:' for coordinate tasks."

            coordinates, answer = answer.split(". Answer: ")
            answer = int(answer.strip())

            coordinates += ". Answer: "  # add back the split part

            answers.append(answer)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": (
                                f"{self.instruction} {question}"
                                if self.instruction
                                else question
                            ),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": coordinates,
                        }
                    ],
                },
            ]

            conversations.append(conversation)

        # remove closing token
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )[:-1].replace("<|im_end|>", "")
            for msg in conversations
        ]

        image_inputs, video_inputs = process_vision_info(conversations)  # type: ignore
        input_ids = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return input_ids, texts, answers, sample_ids


class TrainCollator:
    def __init__(
        self,
        processor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):

        questions = []
        conversations = []
        conversation_prompts = []
        texts = []
        answers = []
        sample_ids = []

        assert len(batch) == 1, "Batch size greater than 1 not supported for training."

        for image, question, answer, sample_id in batch:
            questions.append(question)
            answers.append(answer)
            sample_ids.append(sample_id)

            full_conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": (
                                f"{self.instruction} {question}"
                                if self.instruction
                                else question
                            ),
                        },
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            conversation_prompt = [full_conversation[0]]

            conversations.append(full_conversation)
            conversation_prompts.append(conversation_prompt)

        # add assistant prompt (will ignore it in training)
        text_prompts = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            for conv in conversation_prompts
        ]
        # -1 remove extra space/empty character at the end
        texts = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )[:-1]
            for conv in conversations
        ]

        image_inputs, video_inputs = process_vision_info(  # type: ignore
            conversation_prompts
        )
        prompt_ids = self.processor(
            text=text_prompts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        image_inputs, video_inputs = process_vision_info(conversations)  # type: ignore
        input_ids = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = input_ids.input_ids.detach().clone()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # ignore image and question part
        prompts_shape = prompt_ids.input_ids.shape
        labels[:, : prompts_shape[1]] = -100

        return input_ids, labels
