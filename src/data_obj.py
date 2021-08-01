class InputFeature(object):
    """
        A single set of features of data.
    """

    def __init__(self,
                 passage_input_ids,
                 query_input_ids,
                 passage_input_mask,
                 query_input_mask,
                 passage_segment_ids,
                 query_segment_ids,
                 query_id,
                 passage_id,
                 query_event_start=None,
                 query_event_end=None,
                 query_end_bound=None,
                 passage_event_start=None,
                 passage_event_end=None,
                 passage_end_bound=None,
                 is_positive=None):

        self.passage_input_ids = passage_input_ids
        self.query_input_ids = query_input_ids
        self.passage_input_mask = passage_input_mask
        self.query_input_mask = query_input_mask
        self.passage_segment_ids = passage_segment_ids
        self.query_segment_ids = query_segment_ids
        self.query_id = query_id
        self.passage_id = passage_id
        self.query_event_start = query_event_start
        self.query_event_end = query_event_end
        self.query_end_bound = query_end_bound
        self.passage_event_start = passage_event_start
        self.passage_event_end = passage_event_end
        self.passage_end_bound = passage_end_bound
        self.is_positive = is_positive


class EvaluationObject(object):
    def __init__(self,
                 start_label,
                 end_label,
                 start_pred,
                 end_pred,
                 passage_bound,
                 query_event_start,
                 query_event_end):
        self.start_label = start_label
        self.end_label = end_label
        self.start_pred = start_pred
        self.end_pred = end_pred
        self.passage_bound = passage_bound
        self.query_event_start = query_event_start
        self.query_event_end = query_event_end
