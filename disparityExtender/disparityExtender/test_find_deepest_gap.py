import unittest
import numpy as np

class Scanner:
    def find_deepest_gap(self, ranges):
        """
        Finds the “deepest” gap in the scan by locating the max range
        then taking the contiguous region ≥90% of that max, and
        returning its midpoint index.
        """
        # 1. Find the index of the maximum
        best = ranges.argmax()
        thresh = ranges[best] * 0.9

        # 2. Mask all values ≥ threshold
        mask = ranges >= thresh

        # 3. Compute where runs of True start/end via diff
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends   = np.where(diff == -1)[0]

        # 4. Account for runs touching the array’s ends
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends   = np.r_[ends, mask.size - 1]

        # 5. Locate the run that contains `best`
        run_idx = np.nonzero((starts <= best) & (ends >= best))[0][0]
        left, right = starts[run_idx], ends[run_idx]

        # 6. Return the midpoint
        return (left + right) // 2

        ##############################################################
        # if len(ranges) == 0:
        #     raise ValueError("Empty range array")

        # best_index = np.argmax(ranges)
        # threshold = 0.95 * ranges[best_index]
        # left = best_index
        # right = best_index
        # while left > 0 and ranges[left - 1] >= threshold:
        #     left -= 1
        # while right < len(ranges) - 1 and ranges[right + 1] >= threshold:
        #     right += 1
        # middle = (left + right) // 2
        # return middle

class TestFindDeepestGap(unittest.TestCase):
    def setUp(self):
        self.scanner = Scanner()

    def runTestCase(self, test_name, ranges, expected_output):
        try:
            result = self.scanner.find_deepest_gap(np.array(ranges))
            assert result == expected_output, f"Expected {expected_output}, got {result}"
            print(f"[PASS] {test_name}")
        except AssertionError as e:
            print(f"[FAIL] {test_name} - {e}")
        except Exception as e:
            print(f"[ERROR] {test_name} - {type(e).__name__}: {e}")

    def test_all_cases(self):
        self.runTestCase("Single Peak", [1, 2, 3, 10, 10, 10, 3, 2, 1], 4)
        self.runTestCase("Flat Max", [1, 2, 10, 10, 10, 2, 1], 3)
        self.runTestCase("Multiple Peaks", [1, 2, 10, 2, 1, 3, 10, 3, 1], 2)
        self.runTestCase("All Same", [5, 5, 5, 5, 5], 2)
        self.runTestCase("Custom", [3, 3, 4, 5, 2, 3, 1, 4, 4, 8, 9, 10, 10, 9, 8, 4], 11)
        self.runTestCase("Dropoff Below Threshold", [1, 1, 10, 1, 1], 2)
        self.runTestCase("Empty Array", [], None)  # Will error out

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
