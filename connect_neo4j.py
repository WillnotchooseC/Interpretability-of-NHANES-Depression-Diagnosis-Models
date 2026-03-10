"""
Neo4j 数据库连接模块
用于将本项目的问题分解结果与 Neo4j 图数据库交互
"""

import os
import json
from neo4j import GraphDatabase


# 连接配置 — 请根据你的本地 Neo4j 实例修改
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


class Neo4jConnection:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD, database=NEO4J_DATABASE):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def verify_connection(self):
        """验证数据库连接是否成功"""
        with self.driver.session(database=self.database) as session:
            result = session.run("RETURN 1 AS connected")
            record = result.single()
            return record["connected"] == 1

    def run_query(self, query, parameters=None):
        """执行任意 Cypher 查询"""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def store_decomposition(self, original_question, sub_questions):
        """
        将问题分解结果存入 Neo4j

        参数:
            original_question: 原始复杂问题
            sub_questions: 子问题列表，每项包含 {"step", "description", "sub_question", "type"}
        """
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._create_decomposition, original_question, sub_questions)

    @staticmethod
    def _create_decomposition(tx, original_question, sub_questions):
        # 创建原始问题节点
        tx.run(
            "MERGE (q:Question {text: $text}) SET q.is_complex = true",
            text=original_question,
        )
        # 创建子问题节点和关系
        for sq in sub_questions:
            tx.run(
                """
                MATCH (q:Question {text: $original})
                MERGE (sq:SubQuestion {text: $sub_text})
                SET sq.type = $type, sq.step = $step, sq.description = $desc
                MERGE (q)-[:DECOMPOSED_INTO {step: $step}]->(sq)
                """,
                original=original_question,
                sub_text=sq["sub_question"],
                type=sq.get("type", "unknown"),
                step=sq.get("step", 0),
                desc=sq.get("description", ""),
            )

    def load_predictions(self, predictions_file):
        """从 evaluate.py 的预测结果文件批量导入"""
        with open(predictions_file, "r") as f:
            predictions = json.load(f)

        count = 0
        for item in predictions:
            input_q = item.get("input", "")
            sub_questions = self._parse_output(item.get("predicted", ""))
            if input_q and sub_questions:
                self.store_decomposition(input_q, sub_questions)
                count += 1
        return count

    @staticmethod
    def _parse_output(output_text):
        """解析模型输出格式为结构化子问题列表"""
        sub_questions = []
        lines = output_text.strip().split("\n")
        current_step = {}
        for line in lines:
            line = line.strip()
            if line.startswith("Step "):
                parts = line.split(":", 1)
                step_num = parts[0].replace("Step ", "").strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                current_step = {"step": int(step_num) if step_num.isdigit() else 0, "description": desc}
            elif line.startswith("Sub-question:"):
                parts = line.split("# Type:")
                sq_text = parts[0].replace("Sub-question:", "").strip()
                sq_type = parts[1].strip() if len(parts) > 1 else "unknown"
                current_step["sub_question"] = sq_text
                current_step["type"] = sq_type
                sub_questions.append(current_step)
                current_step = {}
        return sub_questions


if __name__ == "__main__":
    conn = Neo4jConnection()
    try:
        if conn.verify_connection():
            print("Neo4j 连接成功！")
            # 示例：导入预测结果
            # count = conn.load_predictions("nhanes_model/exp1/test_predictions.json")
            # print(f"已导入 {count} 条问题分解记录")
        else:
            print("连接失败")
    except Exception as e:
        print(f"连接错误: {e}")
    finally:
        conn.close()
