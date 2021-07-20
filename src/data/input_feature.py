class InputFeature(object):
    """
        A single set of features of data.
        start & end positions are for the mention bound within the tokenized feature vector
    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 query_id,
                 passage_id,
                 query_start_position=None,
                 query_end_position=None,
                 passage_start_position=None,
                 passage_end_position=None,
                 is_positive=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.query_id = query_id
        self.passage_id = passage_id
        self.query_start_position = query_start_position
        self.query_end_position = query_end_position
        self.passage_start_position = passage_start_position
        self.passage_end_position = passage_end_position
        self.is_positive = is_positive
