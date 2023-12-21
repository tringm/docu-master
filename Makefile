download_evaluation_data: HOTPOT_QA_URL = "https://datasets-server.huggingface.co/rows?dataset=hotpot_qa&config=distractor&split=validation&offset=0&length=100"
download_evaluation_data: OUTPUT_DIR = "tests/resources/evaluation"
download_evaluation_data:
	@curl -X GET $(HOTPOT_QA_URL) | jq '.rows | map(.row)' > $(OUTPUT_DIR)/hotpot_qa.json
