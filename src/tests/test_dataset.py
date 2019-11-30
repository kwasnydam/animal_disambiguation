import unittest

import src.data.dataset as tested_module


class TestProcessText(unittest.TestCase):
    def setUp(self):
        self.object_under_test = tested_module.process_text

    def test_outputs_list_of_sentences_given_string_containing_word_mouse(self):
        test_sentence = 'mouse is an animal.'
        self.assertTrue(type(self.object_under_test(test_sentence)) == list)

    def test_outputs_only_sentences_containing_word_mouse_or_mice(self):
        test_sentences = 'mice and mouse.\nmouse is a small rodent.\ncomputer mouse.\nthere is no valid word here.'
        expected_output = ['mouse and mouse.', 'mouse is a small rodent.', 'computer mouse.']
        self.assertEqual(expected_output, self.object_under_test(test_sentences), 'ERROR: expected = {}, Actual = {}'.
                         format(expected_output, self.object_under_test(test_sentences)))

    def test_changes_to_mouse_for_sentences_containing_mice(self):
        test_sentences = 'mice and mouse.\nmouse is a small rodent.\ncomputer mouse.\nthere is no valid word here.'
        expected_output = ['mouse and mouse.', 'mouse is a small rodent.', 'computer mouse.']
        self.assertEqual(expected_output, self.object_under_test(test_sentences), 'ERROR: expected = {}, Actual = {}'.
                         format(expected_output, self.object_under_test(test_sentences)))


if __name__ == '__main__':
    unittest.main()
