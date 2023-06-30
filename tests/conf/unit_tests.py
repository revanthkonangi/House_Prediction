from ta_lib.regression.custom_transformer import custom_transformer
import os.path as op
import pandas as pd
import unittest
from ta_lib.core.api import create_context, load_dataset, list_datasets
from pprint import pprint

class TestTransformer(unittest.TestCase):
    """A unittest class for checking the working of the custom_transformer."""

    def test_cust_trans(self):
        """It tests the working of the custome transformer.

        Parameters
        ----------
        self : object

        Returns
        ----------
        None

        """

        config_path = op.join('conf', 'config.yml')
        context = create_context(config_path)
        pprint(list_datasets(context))
        housing = load_dataset(context, 'raw/housing')
        ct = custom_transformer()
        ct.fit(housing)
        transformed=ct.transform(housing)
        inv_transformed=ct.inverse_transform(transformed)
        pd.testing.assert_frame_equal(housing, inv_transformed)

if __name__ == "__main__":
    unittest.main()