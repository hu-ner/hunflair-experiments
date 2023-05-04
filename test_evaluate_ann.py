import unittest
import evaluate_ann


class TestNERModelEvaluation(unittest.TestCase):

    def test_exact_match(self):
        true_annotations = {'doc1': [
            ('doc1',  10, 20, 'PERSON'),
            ('doc1',  30, 40, 'ORG')
        ]}
        predicted_annotations = {'doc1': [
            ('doc1', 10, 20, 'PERSON'),
            ('doc1',  30, 40, 'ORG')
        ]}
        result = evaluate_ann.evaluate(true_annotations, predicted_annotations,
                                       match_func=evaluate_ann.exact_match)
        self.assertEqual(result.f_score(), 1.0)

    def test_partial_match_within_offset(self):
        true_annotations = {'doc1': [
            ('doc1', 10, 20, 'PERSON'),
            ('doc1', 30, 40, 'ORG')
        ]}
        predicted_annotations = {'doc1': [
            ('doc1', 12, 20, 'PERSON'),
            ('doc1', 30, 42, 'ORG')
        ]}
        result = evaluate_ann.evaluate(true_annotations, predicted_annotations, match_func=evaluate_ann.partial_match(2))
        self.assertEqual(result.f_score(), 1.0)
        result = evaluate_ann.evaluate(true_annotations, predicted_annotations, match_func=evaluate_ann.partial_match(1))
        self.assertEqual(result.f_score(), 0.0)

    def test_no_predicted_annotations(self):
        true_annotations = {'doc1': [
            ('doc1', 10, 20, 'PERSON'),
            ('doc1', 30, 40, 'ORG')
        ]}
        predicted_annotations = {'doc1': []}
        result = evaluate_ann.evaluate(true_annotations, predicted_annotations,
                                       match_func=evaluate_ann.partial_match(2))
        self.assertAlmostEqual(result.f_score(), 0.0, places=2)

    def test_one_correct_one_false_positive(self):
        true_annotations = {'doc1': [
            ('doc1', 10, 20, 'PERSON')
        ]}
        predicted_annotations = {'doc1': [
            ('doc1', 10, 20, 'PERSON'),
            ('doc1', 30, 40, 'ORG')
        ]}
        result = evaluate_ann.evaluate(true_annotations, predicted_annotations,
                                       match_func=evaluate_ann.partial_match(2))
        self.assertAlmostEqual(result.f_score(), 0.67, places=2)

    def test_one_correct_one_false_negative(self):
        true_annotations = {'doc1': [
            ('doc1', 10, 20, 'PERSON'),
            ('doc1', 30, 40, 'ORG')
        ]}
        predicted_annotations = {'doc1': [
            ('doc1', 10, 20, 'PERSON')
        ]}
        result = evaluate_ann.evaluate(true_annotations, predicted_annotations,
                                       match_func=evaluate_ann.partial_match(2))
        self.assertAlmostEqual(result.f_score(), 0.67, places=2)

    def test_no_match_outside_threshold(self):
        true_annotations = {'doc1': [
            ('doc1', 10, 20, 'PERSON'),
        ]}
        predicted_annotations = {'doc1': [
            ('doc1', 15, 18, 'PERSON')
        ]}
        result = evaluate_ann.evaluate(true_annotations, predicted_annotations,
                                       match_func=evaluate_ann.partial_match(2))
        self.assertAlmostEqual(result.f_score(), 0.0, places=2)

    def test_no_match_different_docs(self):
        true_annotations = {'doc1': [
            ('doc1',  10, 20, 'PERSON'),
        ]}
        predicted_annotations = {'doc1': [
            ('doc2', 10, 20, 'PERSON'),
        ]}
        result = evaluate_ann.evaluate(true_annotations, predicted_annotations,
                                        match_func=evaluate_ann.partial_match(2))
        self.assertAlmostEqual(result.f_score(), 0.0, places=2)

    def test_no_match_different_types(self):
        true_annotations = {'doc1': [
            ('doc1',  10, 20, 'PERSON'),
        ]}
        predicted_annotations = {'doc1': [
            ('doc1', 10, 20, 'ORG'),
        ]}
        result = evaluate_ann.evaluate(true_annotations, predicted_annotations,
                                        match_func=evaluate_ann.partial_match(2))
        self.assertAlmostEqual(result.f_score(), 0.0, places=2)
        
    def test_no_match_substring(self):
        true_annotations = {'doc1': [
                ('doc1', 10, 20, 'PERSON'),
        ]}
        predicted_annotations = {'doc1': [
                ('doc1', 10, 19, 'PERSON'),
        ]}
        result = evaluate_ann.evaluate(
            true_annotations,
            predicted_annotations,
            match_func=evaluate_ann.partial_match(0),
        )
        self.assertAlmostEqual(result.f_score(), 0.0, places=2)

    def test_no_match_true_substring(self):
        true_annotations = {'doc1': [
                ('doc1', 10, 20, 'PERSON'),
        ]}
        predicted_annotations = {'doc1': [
                ('doc1', 12, 19, 'PERSON'),
        ]}
        result = evaluate_ann.evaluate(
            true_annotations,
            predicted_annotations,
            match_func=evaluate_ann.partial_match(0),
        )
        self.assertAlmostEqual(result.f_score(), 0.0, places=2)

    def test_no_match_superstring(self):
        true_annotations = {'doc1': [
                ('doc1', 10, 20, 'PERSON'),
        ]}
        predicted_annotations = {'doc1': [
                ('doc1', 10, 25, 'PERSON'),
        ]}
        result = evaluate_ann.evaluate(
            true_annotations,
            predicted_annotations,
            match_func=evaluate_ann.partial_match(0),
        )
        self.assertAlmostEqual(result.f_score(), 0.0, places=2)

    def test_no_match_true_superstring(self):
        true_annotations = {'doc1': [
                ('doc1', 10, 20, 'PERSON'),
        ]}
        predicted_annotations = {'doc1': [
                ('doc1', 8, 25, 'PERSON'),
        ]}
        result = evaluate_ann.evaluate(
            true_annotations,
            predicted_annotations,
            match_func=evaluate_ann.partial_match(0),
        )
        self.assertAlmostEqual(result.f_score(), 0.0, places=2)

    def test_no_match_shifted_string(self):
        true_annotations = {'doc1': [
                ('doc1', 10, 20, 'PERSON'),
        ]}
        predicted_annotations = {'doc1': [
                ('doc1', 13, 28, 'PERSON'),
        ]}
        result = evaluate_ann.evaluate(
            true_annotations,
            predicted_annotations,
            match_func=evaluate_ann.partial_match(0),
        )
        self.assertAlmostEqual(result.f_score(), 0.0, places=2)        


if __name__ == '__main__':
    unittest.main()
