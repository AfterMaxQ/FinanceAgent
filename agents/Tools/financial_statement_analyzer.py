import requests
import pandas as pd
import time
from typing import Dict, Any, Optional

# --- SEC API 辅助函数 ---

# 建议将User-Agent放入配置文件中
USER_AGENT = "PersonalQuantProject/zhang.wei.123@email.com"

# 重试配置
MAX_RETRIES = 3
RETRY_WAIT_BASE = 2  # 基础等待时间（秒）


def get_cik_by_ticker(ticker: str) -> str:
    """通过股票代码查询10位CIK编码。支持重试机制。"""
    url = "https://www.sec.gov/include/ticker.txt"
    headers = {"User-Agent": USER_AGENT}
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            for line in response.text.splitlines():
                if line:
                    ticker_lower, cik = line.split("\t")
                    if ticker_lower == ticker.lower():
                        return cik.zfill(10)
            raise ValueError(f"未找到 {ticker} 对应的CIK编码")
            
        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt < MAX_RETRIES:
                wait_time = RETRY_WAIT_BASE * attempt
                time.sleep(wait_time)
                continue
            else:
                raise ValueError(f"获取 {ticker} 的CIK编码失败（已重试{MAX_RETRIES}次）: {str(e)}")
    
    raise ValueError(f"未找到 {ticker} 对应的CIK编码")


def get_company_facts(cik: str) -> Dict[str, Any]:
    """获取公司的所有XBRL财务事实数据。支持重试机制。"""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": USER_AGENT}
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            time.sleep(1)  # 遵守SEC每秒最多1个请求的规定
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                wait_time = RETRY_WAIT_BASE * attempt
                time.sleep(wait_time)
                continue
            else:
                raise requests.exceptions.RequestException(
                    f"获取CIK {cik} 的财务数据失败（已重试{MAX_RETRIES}次）: {str(e)}"
                )
    
    raise requests.exceptions.RequestException(f"获取CIK {cik} 的财务数据失败")


def _get_latest_annual_fact_value(facts: Dict, indicator_key: str) -> Optional[Dict]:
    """从facts中提取指定指标最新的年度(10-K)数据点的值和期间。"""
    try:
        units = facts["facts"]["us-gaap"][indicator_key]["units"]["USD"]
        annual_facts = [item for item in units if item.get("form") in ["10-K", "10-K/A"]]
        if not annual_facts:
            return None
        latest_fact = sorted(annual_facts, key=lambda x: x["end"], reverse=True)[0]
        return {"value": latest_fact["val"], "period": latest_fact["end"]}
    except (KeyError, IndexError):
        return None


# --- 核心分析工具类  ---

class FinancialStatementAnalyzer:
    """
    使用SEC API分析公司最新财报，返回包含丰富原始数据的结构化分析报告。
    """

    def analyze_latest_filings(self, ticker: str) -> Dict[str, Any]:
        """
        执行完整的财报分析流程，返回详细的原始数据和分析。
        """
        try:
            cik = get_cik_by_ticker(ticker)
            facts = get_company_facts(cik)

            # --- 1. 提取三大报表的核心原始数据 ---

            # 利润表 (Income Statement)
            revenue_data = _get_latest_annual_fact_value(facts, "Revenues")
            if not revenue_data:
                return {"error": "无法从SEC获取最核心的年度'营业收入'数据。"}

            cost_of_revenue = _get_latest_annual_fact_value(facts, "CostOfRevenue")
            gross_profit = _get_latest_annual_fact_value(facts, "GrossProfit")
            op_income = _get_latest_annual_fact_value(facts, "OperatingIncomeLoss")
            net_income = _get_latest_annual_fact_value(facts, "NetIncomeLoss")

            # 资产负债表 (Balance Sheet)
            assets = _get_latest_annual_fact_value(facts, "Assets")
            current_assets = _get_latest_annual_fact_value(facts, "AssetsCurrent")
            liabilities = _get_latest_annual_fact_value(facts, "Liabilities")
            current_liabilities = _get_latest_annual_fact_value(facts, "LiabilitiesCurrent")
            equity = _get_latest_annual_fact_value(facts, "StockholdersEquity")

            # 现金流量表 (Cash Flow Statement)
            op_cash_flow = _get_latest_annual_fact_value(facts, "NetCashProvidedByUsedInOperatingActivities")
            invest_cash_flow = _get_latest_annual_fact_value(facts, "NetCashProvidedByUsedInInvestingActivities")
            finance_cash_flow = _get_latest_annual_fact_value(facts, "NetCashProvidedByUsedInFinancingActivities")
            capex = _get_latest_annual_fact_value(facts, "PaymentsToAcquirePropertyPlantAndEquipment")

            # --- 2. 构建包含原始数据的字典 ---

            # 使用 .get('value', 0) 来安全地处理可能缺失的非核心数据
            income_metrics = {
                "revenue": revenue_data['value'],
                "cost_of_revenue": cost_of_revenue.get('value', 0) if cost_of_revenue else 0,
                "gross_profit": gross_profit.get('value', 0) if gross_profit else 0,
                "operating_income": op_income.get('value', 0) if op_income else 0,
                "net_income": net_income.get('value', 0) if net_income else 0,
            }

            balance_sheet_metrics = {
                "total_assets": assets.get('value', 0) if assets else 0,
                "current_assets": current_assets.get('value', 0) if current_assets else 0,
                "total_liabilities": liabilities.get('value', 0) if liabilities else 0,
                "current_liabilities": current_liabilities.get('value', 0) if current_liabilities else 0,
                "stockholders_equity": equity.get('value', 0) if equity else 0,
            }

            cash_flow_metrics = {
                "operating_cash_flow": op_cash_flow.get('value', 0) if op_cash_flow else 0,
                "investing_cash_flow": invest_cash_flow.get('value', 0) if invest_cash_flow else 0,
                "financing_cash_flow": finance_cash_flow.get('value', 0) if finance_cash_flow else 0,
                "capital_expenditure": -(capex.get('value', 0)) if capex else 0,  # Capex通常为负
            }

            # --- 3. 计算比率并生成分析 ---

            # 利润表比率
            rev = income_metrics["revenue"]
            gp = income_metrics["gross_profit"]
            ni = income_metrics["net_income"]
            gross_margin = (gp / rev) * 100 if rev != 0 else 0
            net_margin = (ni / rev) * 100 if rev != 0 else 0

            # 资产负债表比率
            ca = balance_sheet_metrics["current_assets"]
            cl = balance_sheet_metrics["current_liabilities"]
            liab = balance_sheet_metrics["total_liabilities"]
            eq = balance_sheet_metrics["stockholders_equity"]
            current_ratio = ca / cl if cl != 0 else 0
            debt_to_equity = liab / eq if eq != 0 else float('inf')

            # 现金流量表比率
            ocf = cash_flow_metrics["operating_cash_flow"]
            fcf = ocf + cash_flow_metrics["capital_expenditure"]  # capex已经是负数
            ocf_to_ni_ratio = ocf / ni if ni != 0 else 0

            # 生成定性分析
            income_take = self._generate_income_take(gross_margin, net_margin)
            balance_sheet_take = self._generate_balance_sheet_take(current_ratio, debt_to_equity)
            cash_flow_take = self._generate_cash_flow_take(ocf, fcf, ocf_to_ni_ratio)

            # --- 4. 组装最终的、信息丰富的报告 ---

            report = {
                "ticker": ticker,
                "company_name": facts.get("entityName", ""),
                "report_period": revenue_data['period'],
                "income_statement_analysis": {
                    "title": "利润表分析 (盈利能力)",
                    "key_metrics": income_metrics,
                    "ratios_and_analysis": {
                        "gross_margin_percent": round(gross_margin, 2),
                        "net_profit_margin_percent": round(net_margin, 2),
                        "analyst_take": income_take
                    }
                },
                "balance_sheet_analysis": {
                    "title": "资产负债表分析 (财务健康状况)",
                    "key_metrics": balance_sheet_metrics,
                    "ratios_and_analysis": {
                        "current_ratio": round(current_ratio, 2),
                        "debt_to_equity_ratio": round(debt_to_equity, 2),
                        "analyst_take": balance_sheet_take
                    }
                },
                "cash_flow_analysis": {
                    "title": "现金流量表分析 (造血能力)",
                    "key_metrics": cash_flow_metrics,
                    "ratios_and_analysis": {
                        "free_cash_flow": fcf,
                        "ocf_to_net_income_ratio": round(ocf_to_ni_ratio, 2),
                        "analyst_take": cash_flow_take
                    }
                },
                "overall_summary": self._generate_overall_summary(income_take, balance_sheet_take, cash_flow_take)
            }
            return report

        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if "503" in error_msg or "Service Unavailable" in error_msg:
                return {
                    "error": (
                        f"SEC API服务暂时不可用（503错误）。"
                        f"这可能是由于SEC服务器维护或请求频率限制。"
                        f"建议稍后重试，或直接访问 https://www.sec.gov/edgar/searchedgar/companysearch.html 查询 {ticker} 的财报。"
                        f"原始错误: {error_msg}"
                    )
                }
            elif "404" in error_msg or "Not Found" in error_msg:
                return {
                    "error": (
                        f"未找到 {ticker} 的财报数据。"
                        f"请确认股票代码正确，或该公司可能未在SEC注册。"
                        f"原始错误: {error_msg}"
                    )
                }
            else:
                return {
                    "error": (
                        f"获取SEC财报数据时发生网络错误: {error_msg}。"
                        f"请检查网络连接或稍后重试。"
                    )
                }
        except ValueError as e:
            return {"error": f"股票代码或CIK查询错误: {str(e)}"}
        except Exception as e:
            return {"error": f"分析财报时发生未知错误: {str(e)}"}

    # 定性分析的辅助函数保持不变
    def _generate_income_take(self, gross_margin, net_margin):
        if net_margin > 20: return "盈利能力极强，净利率非常高，护城河显著。"
        if net_margin > 10: return "盈利能力良好，具有健康的利润空间。"
        if net_margin > 0: return "公司实现盈利，但利润空间较窄，需关注成本控制。"
        return "公司处于亏损状态，盈利能力是当前面临的主要挑战。"

    def _generate_balance_sheet_take(self, current_ratio, debt_to_equity):
        health = []
        if current_ratio > 2:
            health.append("短期偿债能力非常强。")
        elif current_ratio > 1.2:
            health.append("短期偿债能力良好。")
        else:
            health.append("短期流动性需关注，存在一定压力。")

        if debt_to_equity < 0.6:
            health.append("财务杠杆低，结构非常稳健。")
        elif debt_to_equity < 1.5:
            health.append("财务杠杆适中。")
        else:
            health.append("财务杠杆较高，依赖债务驱动，需关注长期债务风险。")
        return " ".join(health)

    def _generate_cash_flow_take(self, ocf, fcf, ocf_to_ni):
        cash_flow = []
        if ocf > 0 and fcf > 0:
            cash_flow.append("经营活动和自由现金流均为正，公司具备强大的自我造血能力。")
            if ocf_to_ni > 1.2:
                cash_flow.append("盈利质量非常高，收到的现金远超账面利润。")
            elif ocf_to_ni > 0.8:
                cash_flow.append("盈利质量健康，现金转化能力良好。")
            else:
                cash_flow.append("盈利质量一般，需要关注部分利润的现金转化情况。")
        elif ocf > 0 and fcf <= 0:
            cash_flow.append("经营现金流为正，但由于大量投资，自由现金流为负，公司处于高速扩张期。")
        else:
            cash_flow.append("经营现金流为负，这是危险信号，表明主营业务正在消耗现金。")
        return " ".join(cash_flow)

    def _generate_overall_summary(self, income_take, balance_sheet_take, cash_flow_take):
        score = 0
        if "极强" in income_take or "良好" in income_take: score += 1
        if "亏损" in income_take: score -= 1
        if "稳健" in balance_sheet_take or "良好" in balance_sheet_take: score += 1
        if "风险" in balance_sheet_take or "压力" in balance_sheet_take: score -= 1
        if "强大" in cash_flow_take or "健康" in cash_flow_take: score += 1
        if "危险信号" in cash_flow_take: score -= 1

        if score >= 3:
            rating = "Excellent (优秀)"
        elif score >= 1:
            rating = "Solid (扎实)"
        elif score == 0:
            rating = "Neutral (中性)"
        else:
            rating = "Caution Needed (需警惕)"

        return {
            "title": "综合评估",
            "long_term_value_rating": rating,
            "summary_text": f"盈利能力: {income_take} 财务健康: {balance_sheet_take} 现金流: {cash_flow_take}"
        }