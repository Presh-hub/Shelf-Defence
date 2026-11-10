import unittest

from src.input_validation import route_request, validate_request


PREPROCESSING_ID = "preprocessing-v1"


def make_request(**changes):
    request = {
        "request_id": "test-001",
        "input_ref": "data/images/test.jpg",
        "format": "jpg",
        "width": 800,
        "height": 800,
        "source": "shelf_camera"
    }
    request.update(changes)
    return request


class InputValidationTests(unittest.TestCase):

    def test_error_overrides_warning(self):
        request = make_request(
            format="bmp",
            width=400,
            height=400
        )

        result = validate_request(request, PREPROCESSING_ID)

        self.assertEqual(result["combined_status"], "error")
        self.assertEqual(result["action"], "reject_input")

    def test_warning_does_not_call_prediction(self):
        request = make_request(width=400, height=400)
        calls = []

        def fake_predict(request):
            calls.append(request)
            return {"model_inference_ran": True}

        result = route_request(
            request, fake_predict, PREPROCESSING_ID
        )

        self.assertEqual(
            result["validation"]["combined_status"], "warning"
        )
        self.assertEqual(
            result["validation"]["action"], "route_for_review"
        )
        self.assertFalse(result["prediction_callback_called"])
        self.assertFalse(result["prediction_ran"])
        self.assertEqual(calls, [])

    def test_pass_calls_prediction(self):
        request = make_request()
        calls = []

        def fake_predict(request):
            calls.append(request)
            return {
                "model_inference_ran": True,
                "result": "test prediction"
            }

        result = route_request(
            request, fake_predict, PREPROCESSING_ID
        )

        self.assertEqual(
            result["validation"]["combined_status"], "pass"
        )
        self.assertTrue(result["prediction_callback_called"])
        self.assertTrue(result["prediction_ran"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            result["prediction"]["result"], "test prediction"
        )


if __name__ == "__main__":
    unittest.main()
