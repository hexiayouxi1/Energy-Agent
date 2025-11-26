import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import tempfile
import torch

# Darts Imports
from darts import TimeSeries
from darts.models import (
    NBEATSModel,
    TransformerModel,
    TCNModel,
    BlockRNNModel,
    LinearRegressionModel
)
from darts.dataprocessing.transformers import Scaler

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. KNOWLEDGE BASE
# ==============================================================================
EMISSION_FACTORS = {
    'coal_consumption': 0.95,
    'oil_consumption': 0.75,
    'gas_consumption': 0.45,
    'nuclear_consumption': 0.0,
    'hydro_consumption': 0.0,
    'solar_consumption': 0.0,
    'wind_consumption': 0.0,
    'biofuel_consumption': 0.05
}
RENEWABLES = ['hydro_consumption', 'solar_consumption', 'wind_consumption', 'biofuel_consumption']


# ==============================================================================
# 2. DATA ENGINE
# ==============================================================================
class DataEngine:
    def __init__(self):
        self.df = None
        self.countries = []
        self.energy_cols = [
            'coal_consumption', 'oil_consumption', 'gas_consumption',
            'nuclear_consumption', 'hydro_consumption',
            'solar_consumption', 'wind_consumption', 'biofuel_consumption'
        ]
        self.covariates = ['gdp', 'population']

    def load_data(self, file_obj):
        if file_obj is None: return None, "No file", gr.update()
        try:
            df = pd.read_csv(file_obj.name)
            col_map = {'Entity': 'country', 'Country': 'country', 'Year': 'year'}
            df.rename(columns=col_map, inplace=True)
            df.columns = [c.lower() for c in df.columns]

            for target in self.energy_cols + self.covariates:
                if target not in df.columns:
                    for c in df.columns:
                        if target.split('_')[0] in c and ('consumption' in c or target in c):
                            df.rename(columns={c: target}, inplace=True)
                            break

            self.df = df
            self.countries = sorted(df['country'].unique().tolist())
            msg = f"✅ Loaded {len(df)} rows. Countries: {len(self.countries)}"
            return df.head(), msg, gr.Dropdown(choices=self.countries, value="China")
        except Exception as e:
            return None, f"❌ Error: {str(e)}", gr.update()


engine = DataEngine()


# ==============================================================================
# 3. ANALYTICS ENGINE
# ==============================================================================
class AnalyticsEngine:
    @staticmethod
    def calculate_metrics(full_df):
        res = full_df.copy()
        valid_cols = [c for c in engine.energy_cols if c in res.columns]
        res['total_energy'] = res[valid_cols].sum(axis=1)

        ren_cols = [c for c in RENEWABLES if c in res.columns]
        if ren_cols:
            res['renewable_share'] = res[ren_cols].sum(axis=1) / res['total_energy']
        else:
            res['renewable_share'] = 0

        res['total_emissions'] = 0
        for col, factor in EMISSION_FACTORS.items():
            if col in res.columns:
                res['total_emissions'] += res[col] * factor
        return res

    @staticmethod
    def calculate_cagr(start_val, end_val, years):
        if start_val <= 0: return 0.0
        return (end_val / start_val) ** (1 / years) - 1


# ==============================================================================
# 4. AI CORE
# ==============================================================================
class DartsAgent:
    def __init__(self):
        self.scaler = Scaler()

    def run_forecast(self, df, country, model_name, horizon):
        if df is None: raise ValueError("Data not loaded")

        sub = df[df['country'] == country].sort_values('year')
        if len(sub) < 15: raise ValueError(f"History too short (<15 years)")

        targets = [c for c in engine.energy_cols if c in sub.columns]
        covs = [c for c in engine.covariates if c in sub.columns]

        sub[targets + covs] = sub[targets + covs].interpolate().fillna(method='bfill').fillna(0)

        series_target = TimeSeries.from_dataframe(sub, 'year', targets)
        series_cov = TimeSeries.from_dataframe(sub, 'year', covs) if covs else None

        scaler_target = Scaler()
        series_target_scaled = scaler_target.fit_transform(series_target)

        scaler_cov = Scaler() if series_cov else None
        series_cov_scaled = scaler_cov.fit_transform(series_cov) if series_cov else None

        common_params = {
            "input_chunk_length": min(10, len(sub) // 2),
            "output_chunk_length": horizon,
            "n_epochs": 50,  # 快速演示用50，实际可调高
            "random_state": 42,
            "pl_trainer_kwargs": {"accelerator": "gpu" if torch.cuda.is_available() else "cpu"}
        }

        if model_name == "N-BEATS":
            model = NBEATSModel(**common_params)
        elif model_name == "Transformer":
            model = TransformerModel(d_model=64, nhead=4, **common_params)
        elif model_name == "LSTM (BlockRNN)":
            model = BlockRNNModel(model="LSTM", hidden_dim=25, **common_params)
        elif model_name == "TCN":
            model = TCNModel(kernel_size=3, **common_params)
        else:
            lags_past = common_params['input_chunk_length'] if series_cov else None
            model = LinearRegressionModel(
                lags=common_params['input_chunk_length'],
                lags_past_covariates=lags_past,
                output_chunk_length=horizon
            )

        if model_name == "Linear Regression":
            model.fit(series_target_scaled, past_covariates=series_cov_scaled)
        else:
            model.fit(series_target_scaled, past_covariates=series_cov_scaled, verbose=False)

        pred_scaled = model.predict(n=horizon, series=series_target_scaled, past_covariates=series_cov_scaled)
        pred = scaler_target.inverse_transform(pred_scaled)

        def to_pandas(ts):
            v = ts.values()
            if v.ndim == 3: v = v[:, :, 0]
            return pd.DataFrame(v, index=ts.time_index, columns=ts.components)

        hist_df = to_pandas(series_target)
        pred_df = to_pandas(pred)
        pred_df[pred_df < 0] = 0

        return hist_df, pred_df


agent = DartsAgent()


# ==============================================================================
# 5. UI & VISUALIZATION (ADVANCED)
# ==============================================================================
def generate_report(country, hist_df, pred_df_ai):
    """保存 AI 预测的 CSV 到临时文件"""
    full = pd.concat([hist_df, pred_df_ai])
    analyzed = AnalyticsEngine.calculate_metrics(full)

    filename = f"forecast_{country}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv"
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    analyzed.to_csv(file_path)
    return file_path


def gui_run(country, model, horizon, df_state):
    try:
        horizon = int(horizon)

        # --- 1. 双模型运行 (Comparison) ---
        # A. 运行用户选择的模型 (AI / Main)
        hist, pred_ai = agent.run_forecast(df_state, country, model, horizon)

        # B. 运行线性基线 (Linear Baseline) 用于对比
        # 如果用户选的就是线性，那两条线重合
        if model == "Linear Regression":
            pred_lin = pred_ai.copy()
        else:
            _, pred_lin = agent.run_forecast(df_state, country, "Linear Regression", horizon)

        # --- 2. 数据分析 ---
        full_ai = pd.concat([hist, pred_ai])
        full_lin = pd.concat([hist, pred_lin])

        # 计算指标
        metrics_ai = AnalyticsEngine.calculate_metrics(full_ai)
        metrics_lin = AnalyticsEngine.calculate_metrics(full_lin)

        # --- 3. 生成高级图表 ---
        # 使用 specs 创建双轴子图
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"colspan": 2}, None],
                [{"type": "xy"}, {"secondary_y": True}]  # Row 2 Col 2 开启双轴
            ],
            subplot_titles=(
                f"A. Energy Mix Forecast ({model}): Non-Linear Transition",
                "B. Adoption Gap: Renewable Share (AI vs Linear)",
                "C. Environmental Impact: Energy-Emissions Decoupling"
            ),
            vertical_spacing=0.15
        )

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        # === Plot 1: Stacked Area (Main AI Forecast) ===
        for i, col in enumerate(hist.columns):
            full_idx = list(hist.index) + list(pred_ai.index)
            full_val = list(hist[col].values) + list(pred_ai[col].values)

            fig.add_trace(go.Scatter(
                x=full_idx, y=full_val, name=col.split('_')[0].title(),
                stackgroup='one', mode='none', fillcolor=colors[i % len(colors)]
            ), row=1, col=1)

        fig.add_vline(x=hist.index[-1], line_dash="dash", line_color="white", row=1, col=1,
                      annotation_text="Forecast Start")

        # === Plot 2: Adoption Gap (Linear vs AI) ===
        # 只取历史最后5年 + 预测期
        start_plot_yr = hist.index[-5]

        plot_ai = metrics_ai.loc[start_plot_yr:]
        plot_lin = metrics_lin.loc[start_plot_yr:]

        # 1. Linear Baseline (Grey)
        fig.add_trace(go.Scatter(
            x=plot_lin.index, y=plot_lin['renewable_share'] * 100,
            name="Linear Baseline (BAU)", line=dict(color='gray', width=2, dash='dot')
        ), row=2, col=1)

        # 2. AI Optimized (Green)
        fig.add_trace(go.Scatter(
            x=plot_ai.index, y=plot_ai['renewable_share'] * 100,
            name=f"AI Forecast ({model})", line=dict(color='#00ff00', width=3),
            fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)'  # Fill gap
        ), row=2, col=1)

        # 3. Target
        fig.add_hline(y=50, line_dash="solid", line_color="#ff4444", annotation_text="Net Zero (50%)", row=2, col=1)

        # === Plot 3: Decoupling (Energy vs Emissions) ===
        plot_decouple = metrics_ai.loc[start_plot_yr:]
        plot_decouple_lin = metrics_lin.loc[start_plot_yr:]

        # Axis 1: Total Energy (Blue)
        fig.add_trace(go.Scatter(
            x=plot_decouple.index, y=plot_decouple['total_energy'],
            name="Total Energy Demand", line=dict(color='#2980b9', width=3)
        ), row=2, col=2, secondary_y=False)

        # Axis 2: Emissions (Red) - Comparison
        # Linear Emissions (Faint)
        fig.add_trace(go.Scatter(
            x=plot_decouple_lin.index, y=plot_decouple_lin['total_emissions'],
            name="Baseline Emissions", line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ), row=2, col=2, secondary_y=True)

        # AI Emissions (Solid Red)
        fig.add_trace(go.Scatter(
            x=plot_decouple.index, y=plot_decouple['total_emissions'],
            name="AI Projected Emissions", line=dict(color='#c0392b', width=3),
            fill='tonexty', fillcolor='rgba(46, 204, 113, 0.2)'  # Green fill for Abatement Wedge
        ), row=2, col=2, secondary_y=True)

        # 布局美化
        fig.update_layout(
            template="plotly_dark", height=800,
            paper_bgcolor="#1f2937", plot_bgcolor="#1f2937",
            font=dict(color="white"), hovermode="x unified",
            legend=dict(orientation="h", y=-0.1)
        )
        fig.update_yaxes(title_text="TWh", row=1, col=1)
        fig.update_yaxes(title_text="Share (%)", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="Total Energy (TWh)", row=2, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Emissions (Mt CO2)", row=2, col=2, secondary_y=True)

        # --- 4. 文本摘要 ---
        last_hist = metrics_ai.loc[hist.index[-1]]
        last_pred = metrics_ai.loc[pred_ai.index[-1]]
        last_lin = metrics_lin.loc[pred_lin.index[-1]]

        energy_cagr = AnalyticsEngine.calculate_cagr(last_hist['total_energy'], last_pred['total_energy'], horizon)
        co2_cagr = AnalyticsEngine.calculate_cagr(last_hist['total_emissions'], last_pred['total_emissions'], horizon)

        # 计算 Adoption Gap
        share_gap = (last_pred['renewable_share'] - last_lin['renewable_share']) * 100

        summary = f"""
        ### 🧠 Strategic Insights (AI vs Linear Baseline)
        - **Adoption Gap**: The {model} model predicts a Renewable Share of **{last_pred['renewable_share']:.1%}** by {pred_ai.index[-1]}, which is **{share_gap:+.1f}%** compared to the Linear Baseline.
        - **Decoupling Status**: While Energy Demand grows at **{energy_cagr:.1%}** CAGR, CO2 Emissions are changing at **{co2_cagr:.1%}**.
        - **Abatement Wedge**: AI optimization identifies a pathway to reduce emissions to **{last_pred['total_emissions']:.0f} Mt**, compared to **{last_lin['total_emissions']:.0f} Mt** in the BAU scenario.
        """

        csv_path = generate_report(country, hist, pred_ai)
        return fig, summary, csv_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}", None


# ==============================================================================
# 6. APP LAYOUT
# ==============================================================================
custom_css = """
body {background-color: #0e1117}
.gradio-container {font-family: 'IBM Plex Mono', sans-serif;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="slate"), css=custom_css,
               title="Helios Energy Agent") as demo:
    df_state = gr.State(None)

    gr.Markdown("# ⚡ Helios: AI-Powered Energy Strategy Agent")
    gr.Markdown("Comparing **Deep Learning (AI-Optimized)** vs **Linear (BAU)** pathways for Net Zero Transition.")

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 1. Data Injection")
            file_in = gr.File(label="Upload OWID CSV")
            btn_load = gr.Button("🔄 Load Knowledge Base", variant="secondary")

            gr.Markdown("### 2. Strategy Config")
            dd_country = gr.Dropdown(["(Load Data First)"], label="Target Market")
            dd_model = gr.Dropdown(
                ["N-BEATS", "Transformer", "LSTM (BlockRNN)", "TCN", "Linear Regression"],
                value="N-BEATS", label="AI Model (Challenger)"
            )
            sl_hor = gr.Slider(1, 30, value=20, label="Forecast Horizon (Years)")

            btn_run = gr.Button("🚀 Generate Strategy", variant="primary")

            gr.Markdown("### 3. Export")
            file_out = gr.File(label="Download AI Report")

        with gr.Column(scale=3):
            plot = gr.Plot(label="Strategic Dashboard")
            kpi_board = gr.Markdown("### ⏳ Waiting for simulation...")

    btn_load.click(
        engine.load_data, [file_in], [gr.DataFrame(visible=False), kpi_board, dd_country]
    ).then(lambda: engine.df, None, df_state)

    btn_run.click(
        gui_run,
        [dd_country, dd_model, sl_hor, df_state],
        [plot, kpi_board, file_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)