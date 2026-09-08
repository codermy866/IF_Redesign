import unittest
from audit_oct_site_labels import parse_sites


class SiteParsing(unittest.TestCase):
    def test_user_rule_not_case_grade(self):
        self.assertEqual(parse_sites('低级别12')[0],[12])
        self.assertEqual(parse_sites('未发现(低级别1,2)')[0],[1,2])
        self.assertEqual(parse_sites('高危（1、4、12）')[0],[1,4,12])

    def test_negative(self):
        self.assertEqual(parse_sites('未发现')[0],[])

    def test_ranges(self):
        self.assertEqual(parse_sites('疑似3-5,12')[0],[3,4,5,12])

    def test_fail_closed(self):
        for value in ('疑似8,1211','高危14','高危4.5','', '图像畸变无法判读'):
            self.assertIsNone(parse_sites(value)[0])


if __name__=='__main__':unittest.main()
