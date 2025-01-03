{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"},"kaggle":{"accelerator":"none","dataSources":[],"isInternetEnabled":true,"language":"python","sourceType":"script","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code]\nfrom typing import Optional\nfrom pathlib import Path\n\nimport torch\nimport polars as pl\n\nimport os\nimport sys\nimport inspect\n\ncurrentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))\nparentdir = os.path.dirname(currentdir)\nsys.path.insert(0, parentdir)\n\nfrom train.train import ModelRunner\n\n\ndef make_parquet_path(i: int) -> str:\n    \"\"\"Create a string path to a parquet file.\"\"\"\n    return (\n        f\"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/\"\n        f\"partition_id={i}/part-0.parquet\"\n    )\n\n\nclass Evaluator:\n    def __init__(self, ckpt: Path):\n        self.window_size: int = 16\n        self.runner = ModelRunner(window_size=self.window_size)\n\n        self.checkpoint = torch.load(ckpt)\n        self.runner.model.load_state_dict(self.checkpoint[\"state_dict\"])\n\n        self.lags: Optional[pl.DataFrame] = None\n\n    def create_memory_context(self):\n        # Setup the memory context using the full training set and report the mean errors.\n        with torch.no_grad():\n            self.runner.model.eval()\n            self.runner.model.reset_memory()\n            mae, _ = self.runner.run_epoch(\n                dataloader=self.runner.val_dataloader,\n                train_seq_len=self.runner.train_seq_len,\n            )\n\n        print(\"MAE: \", mae)\n\n    def predict(self, test: pl.DataFrame, lags: Optional[pl.DataFrame] = None) -> pl.DataFrame:\n        if lags is not None:\n            self.lags = lags\n\n        # Lagged Responders. Date IDs are advanced by 1 to match them with their associated feature.\n        lagged_df: torch.Tensor = torch.from_numpy(test.fill_nan(0.0).fill_null(0.0).to_numpy())\n        date_ids: torch.Tensor = lagged_df[[0]].int() - 1\n        time_ids: torch.Tensor = lagged_df[[1]].int()\n        symbol_ids: torch.Tensor = lagged_df[[2]].int()\n        responders: torch.Tensor = lagged_df[3:].float()\n\n        # Current date and time\n        current_df: torch.Tensor = torch.from_numpy(test.fill_nan(0.0).fill_null(0.0).to_numpy())\n        date_ids = torch.vstack(date_ids, current_df[[1]])\n        time_ids = torch.vstack(time_ids, current_df[[2]])\n        symbol_ids = torch.vstack(symbol_ids, current_df[[3]])\n        features: torch.Tensor = current_df[6:]\n\n        predictions: torch.Tensor = self.runner.model.forward(\n            date_ids=date_ids.cuda().unsqueeze(0),\n            symbol_ids=symbol_ids.cuda().unsqueeze(0),\n            time_ids=time_ids.cuda().unsqueeze(0),\n            features=features.cuda().unsqueeze(0),\n            responders=responders.cuda().unsqueeze(0),\n        )\n\n        return test.select(\"row_id\").with_columns(pl.Series(\"responder_6\", predictions[0, :, 6]))\n","metadata":{"_uuid":"41a127f3-8213-4127-bee8-03ccfeb7f939","_cell_guid":"c522f65e-b859-468d-8c63-cb2392e26272","trusted":true,"collapsed":false,"jupyter":{"outputs_hidden":false}},"outputs":[],"execution_count":null}]}