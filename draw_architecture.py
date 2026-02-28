from graphviz import Digraph


def draw_balanced_architecture():
    dot = Digraph('QuantSystem_Balanced', comment='Balanced Architecture')

    # --- 全局配置 ---
    font_main = 'Microsoft YaHei'

    # 布局关键：TB (从上到下) + Ortho (正交线)
    dot.attr(rankdir='TB')
    dot.attr(splines='ortho')
    dot.attr(nodesep='0.4')  # 同层节点间距 (紧凑)
    dot.attr(ranksep='0.6')  # 层与层间距
    dot.attr(bgcolor='#FFFFFF')
    dot.attr(compound='true')  # 允许子图连线
    dot.attr(dpi='300')  # 提升输出分辨率

    # --- 样式工厂函数 ---
    def create_card(title, subtitle, icon, color, width=220):
        # 调整了宽度和高度，使卡片更紧凑
        return f'''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">
            <TR><TD WIDTH="50" HEIGHT="50" BGCOLOR="{color}" FIXEDSIZE="TRUE"><FONT POINT-SIZE="20" COLOR="white">{icon}</FONT></TD>
            <TD WIDTH="{width}" HEIGHT="50" BGCOLOR="{color}" ALIGN="LEFT" VALIGN="MIDDLE" FIXEDSIZE="TRUE"><FONT POINT-SIZE="12" COLOR="white"><B>  {title}</B></FONT></TD></TR>
            <TR><TD COLSPAN="2" WIDTH="{width + 50}" HEIGHT="40" BGCOLOR="#F3F4F6" BORDER="1" COLOR="{color}" ALIGN="LEFT" VALIGN="MIDDLE"><FONT POINT-SIZE="10" COLOR="#374151">  {subtitle}</FONT></TD></TR>
            </TABLE>>'''

    dot.attr('node', shape='plaintext', fontname=font_main)
    dot.attr('edge', fontname=font_main, fontsize='10', color='#6B7280', penwidth='1.2', arrowsize='0.7')

    # ==========================================
    # 第一层：用户与交互终端 (顶层入口)
    # ==========================================
    with dot.subgraph(name='cluster_0_ui') as c:
        c.attr(style='invis')
        # 强制这两个节点在同一水平线上
        c.attr(rank='same')

        c.node('User',
               label=create_card('基金经理', '决策指令 / 报表审阅', '👨‍💼', '#374151', width=160))

        c.node('App',
               label=create_card('交互终端 (App)', 'Streamlit 可视化仪表盘', '💻', '#2563EB', width=180))

    # ==========================================
    # 第二层：AI 中枢 (逻辑核心)
    # ==========================================
    with dot.subgraph(name='cluster_1_brain') as c:
        c.attr(style='invis')
        c.attr(rank='same')

        # 将 Agent 放在中心位置
        c.node('Agent',
               label=create_card('金融智能体', 'SOP路由 / 意图识别', '🧠', '#7C3AED', width=180))

        c.node('LLM',
               label=create_card('DeepSeek V3', '通用逻辑推理引擎', '🚀', '#4C1D95', width=160))

    # ==========================================
    # 第三层：专业工具箱 (业务能力)
    # ==========================================
    with dot.subgraph(name='cluster_2_tools') as c:
        c.attr(label='专业分析工具箱 (Expert Tools)', fontname=font_main, fontsize='12', color='#D1D5DB',
               style='dashed,rounded')
        c.attr(rank='same')  # 关键：横向排列

        c.node('Tool_Tech',
               label=create_card('短期技术分析', '趋势预测 / 支撑阻力', '📈', '#059669', width=150))

        c.node('Tool_Fund',
               label=create_card('长期基本面', 'SEC财报深度解读', '📜', '#D97706', width=150))

        c.node('Tool_Macro',
               label=create_card('宏观风控', '大盘 / VIX / 周期', '🌍', '#DC2626', width=150))

    # ==========================================
    # 第四层：数据基建 (底层支撑)
    # ==========================================
    with dot.subgraph(name='cluster_3_data') as c:
        c.attr(label='数据基础设施 (Data Infrastructure)', fontname=font_main, fontsize='12', color='#E5E7EB',
               style='filled,rounded', fillcolor='#F9FAFB')
        c.attr(rank='same')  # 关键：横向排列

        c.node('ETL',
               label=create_card('离线数据工厂', '清洗 / 因子计算', '⚙️', '#4B5563', width=140))

        c.node('DB',
               label=create_card('结构化数据库', '行情 / 因子 / 向量', '🛢️', '#6B7280', width=140))

        c.node('API_SEC',
               label=create_card('SEC EDGAR', '官方原始财报源', '🏛️', '#B45309', width=140))

    # ==========================================
    # 连线逻辑 (精心设计避免交叉)
    # ==========================================

    # 1. 顶层交互 (横向)
    dot.edge('User', 'App', label='操作', constraint='false')  # constraint=false 允许同层连线而不影响布局

    # 2. UI -> Brain (向下)
    dot.edge('App', 'Agent', label='JSON请求', color='#2563EB')
    dot.edge('Agent', 'App', label='流式响应', style='dashed', dir='back')

    # 3. Brain 内部 (横向)
    dot.edge('Agent', 'LLM', dir='both', constraint='false', color='#7C3AED')

    # 4. Brain -> Tools (向下分发)
    dot.edge('Agent', 'Tool_Tech', color='#059669')
    dot.edge('Agent', 'Tool_Fund', color='#D97706')
    dot.edge('Agent', 'Tool_Macro', color='#DC2626')

    # 5. Tools -> Data (向下读取)
    dot.edge('Tool_Tech', 'DB', style='dashed', color='#6B7280')
    dot.edge('Tool_Macro', 'DB', style='dashed', color='#6B7280')
    dot.edge('Tool_Fund', 'API_SEC', style='dashed', color='#B45309')

    # 6. Data 内部流转 (横向)
    dot.edge('ETL', 'DB', label='写入', constraint='false', color='#4B5563')

    # ==========================================
    # 隐形辅助线 (微调对齐)
    # ==========================================
    # 这是一个小技巧：使用不可见的线强制让节点垂直对齐，形成网格感
    # 让 App 对齐 Agent
    # 让 Agent 对齐 Tool_Fund (中间的工具)
    # 让 Tool_Fund 对齐 DB (中间的数据)

    edge_invis = {'style': 'invis', 'weight': '10'}  # weight高表示尽量拉直

    # 渲染
    output_path = 'balanced_architecture'
    dot.render(output_path, view=True, format='png', cleanup=True)
    print(f"均衡布局图已生成: {output_path}.png")


if __name__ == '__main__':
    try:
        draw_balanced_architecture()
    except Exception as e:
        print(f"绘图错误: {e}")