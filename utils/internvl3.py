from transformers import BatchEncoding


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

        input_ids = self.processor.apply_chat_template(
            conversations,
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
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
                        },
                    ],
                },
            ]

            conversations.append(conversation)

        input_ids = self.processor.apply_chat_template(
            conversations,
            padding=True,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # remove <|im_end|>\n at the end
        input_ids.input_ids = input_ids.input_ids[:, :-2]
        input_ids.attention_mask = input_ids.attention_mask[:, :-2]

        questions = self.processor.batch_decode(
            input_ids.input_ids, skip_special_tokens=True
        )

        return (
            BatchEncoding(  # required to update the content of input_ids
                data={
                    "input_ids": input_ids.input_ids,
                    "attention_mask": input_ids.attention_mask,
                    "pixel_values": input_ids.pixel_values,
                }
            ),
            questions,
            answers,
            sample_ids,
        )


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
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer,
                        },
                    ],
                },
            ]
            conversation_prompt = [full_conversation[0]]

            conversations.append(full_conversation)
            conversation_prompts.append(conversation_prompt)

        # add assistant prompt (will ignore it in training)
        prompt_ids = self.processor.apply_chat_template(
            conversation_prompts,
            padding=False,  # only one sample in batch
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = self.processor.apply_chat_template(
            conversations,
            padding=False,  # only one sample in batch
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # remove extra \n at the end
        input_ids.input_ids = input_ids.input_ids[:, :-1]
        input_ids.attention_mask = input_ids.attention_mask[:, :-1]

        labels = input_ids.input_ids.detach().clone()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # ignore image and question part
        prompts_shape = prompt_ids.input_ids.shape
        labels[:, : prompts_shape[1]] = -100

        return (
            BatchEncoding(  # required to update the content of input_ids
                data={
                    "input_ids": input_ids.input_ids,
                    "attention_mask": input_ids.attention_mask,
                    "pixel_values": input_ids.pixel_values,
                }
            ),
            labels,
        )
