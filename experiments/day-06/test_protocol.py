import unittest
import common


class ProtocolTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(common.key(' ЁЖ,  пришёл! '), common.key('еж пришел'))

    def test_source_and_target_leakage(self):
        train = [{'source': 'Первый текст', 'target': 'Второй текст'}]
        with self.assertRaises(ValueError):
            common.assert_disjoint(train, [{'source': 'ВТОРОЙ текст!', 'target': 'Другой'}])

    def test_empty_and_contacts_excluded(self):
        self.assertFalse(common.eligible(' ', 'Ответ'))
        self.assertFalse(common.eligible('Напиши test@example.com', 'Ответ'))
        self.assertFalse(common.eligible('Текст <|im_end|>', 'Ответ'))
        self.assertTrue(common.eligible('Исправь ошибку!', 'Исправьте ошибку.'))

    def test_message_roles(self):
        self.assertEqual([m['role'] for m in common.messages('Вход', 'Выход')], ['system', 'user', 'assistant'])
        self.assertEqual(len(common.messages('Вход')), 2)

    def test_toxic_targets(self):
        self.assertTrue(common.toxic_target('Ты полный идиот.'))
        self.assertTrue(common.toxic_target('Это хуйня.'))
        self.assertFalse(common.toxic_target('Застрахуйте поездку.'))
        self.assertFalse(common.toxic_target('Мне не нравится качество.'))


if __name__ == '__main__':
    unittest.main()
