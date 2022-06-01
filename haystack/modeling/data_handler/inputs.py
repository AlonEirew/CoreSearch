from typing import List, Union, Optional


class Question:
    def __init__(self, text: str, uid: str=None):
        self.text = text
        self.uid = uid

    def to_dict(self):
        ret = {"question": self.text,
               "id": self.uid,
               "answers": []}
        return ret


class QAInput:
    def __init__(self, doc_text: str, questions: Union[List[Question], Question], title=Optional[str]):
        self.doc_text = doc_text
        self.title = title
        if type(questions) == Question:
            self.questions = [questions]
        else:
            self.questions = questions #type: ignore

    def to_dict(self):
        questions = [q.to_dict() for q in self.questions]
        ret = {"qas": questions,
               "context": self.doc_text,
               "title": self.title}
        return ret

