import torch


class GenerationCollator:
    def __init__(
        self,
        processor,
        instruction,
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):

        samples = []
        questions = []
        images = []
        answers = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            samples.append(
                f"{self.instruction} {question}" if self.instruction else question
            )
            questions.append(question)
            images.append(image)
            answers.append(answer)
            sample_ids.append(sample_id)

        # MOLMo for now cannot process multiple samples at once, just pop the first item since we will use it with batch_size = 1
        input_ids = self.processor.process(
            text=samples.pop(), images=images, return_tensors="pt", padding=True
        )

        questions = self.processor.tokenizer(
            questions, return_tensors="pt", padding=True
        )

        return input_ids, questions, answers, sample_ids


class TrainCollator:
    def __init__(
        self,
        processor,
        instruction,
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):

        samples = []
        samples_no_answ = []
        questions = []
        images = []
        answers = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            sample_no_answ = (
                f"{self.instruction} {question}" if self.instruction else question
            )
            # manually apply prompt = "User: " + prompt + " Assistant:" (huggingface.co/allenai/Molmo-7B-O-0924/blob/main/preprocessing_molmo.py)
            prompt_no_answ = f"User: {sample_no_answ} Assistant:"
            samples_no_answ.append(prompt_no_answ)
            samples.append(f"{prompt_no_answ} {answer}")
            questions.append(question)
            images.append(image)
            answers.append(answer)
            sample_ids.append(sample_id)

        # MOLMo for now cannot process multiple samples at once, just pop the first item since we will use it with batch_size = 1
        assert len(samples) == 1
        # message_format=None to manually apply template
        input_no_answ = self.processor.process(
            text=samples_no_answ.pop(),
            images=images,
            return_tensors="pt",
            padding=True,
            message_format=None,
        )
        input_ids = self.processor.process(
            text=samples.pop(),
            images=images,
            return_tensors="pt",
            padding=True,
            message_format=None,
        )

        # add batch dimension and convert float32 to bfloat16
        for k, v in input_no_answ.items():
            v_new = v.unsqueeze(0)
            if v.dtype == torch.float32:
                v_new = v_new.type(torch.bfloat16)
            input_no_answ[k] = v_new
        for k, v in input_ids.items():
            v_new = v.unsqueeze(0)
            if v.dtype == torch.float32:
                v_new = v_new.type(torch.bfloat16)
            input_ids[k] = v_new

        questions = self.processor.tokenizer(
            questions, return_tensors="pt", padding=True
        )

        # NO SHIFT
        labels = input_ids["input_ids"].detach().clone()

        # ignore pad
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # ignore image and question
        no_answ_shape = input_no_answ["input_ids"].shape

        # no batch
        labels[: no_answ_shape[0], : no_answ_shape[1]] = -100

        return input_ids, labels
