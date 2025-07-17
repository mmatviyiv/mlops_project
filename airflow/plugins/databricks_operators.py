import json
from airflow.exceptions import AirflowException
from airflow.providers.databricks.hooks.databricks import DatabricksHook
from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator


class DatabricksRunNotebookOperator(DatabricksSubmitRunOperator):
    """
    A custom operator that extends the standard DatabricksSubmitRunOperator to
    submit a run, wait for its completion, and then retrieve and return the
    value from the notebook's `dbutils.notebook.exit()` call.
    """

    def execute(self, context):
        # Call the parent execute method to submit the run and wait for completion.
        # This is a blocking call. The run_id is stored in self.run_id.
        super().execute(context)

        hook = DatabricksHook(self.databricks_conn_id)

        print(f"Parent run {self.run_id} completed. Fetching its details to find the task run ID...")
        run_info = hook._do_api_call(("GET", f"2.1/jobs/runs/get?run_id={self.run_id}"))

        run_tasks = run_info.get("tasks", [])
        if not run_tasks:
            raise AirflowException(f"No tasks found for run_id {self.run_id}. Cannot retrieve output.")

        # Assuming there is only one task as per our DAG definition
        task_run_id = run_tasks[0].get("run_id")
        if not task_run_id:
            raise AirflowException(f"Could not find task_run_id in the run details: {run_info}")

        print(f"Found task run ID: {task_run_id}. Retrieving its output...")
        run_output = hook.get_run_output(task_run_id)

        notebook_output = run_output.get("notebook_output", {})
        result = notebook_output.get("result")
        print(f"Notebook results: {result}")

        result = result.replace("'", '"') if result else "{}"

        return json.loads(result)
