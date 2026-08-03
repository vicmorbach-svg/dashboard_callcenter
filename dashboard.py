import streamlit as st
import polars as pl
import numpy as np
import os
import re
import unicodedata
import gc
import plotly.express as px
from io import BytesIO

# Importações adicionais para a API do GitHub
import requests
import base64
import json
import io
import datetime

# O prefixo para os arquivos de histórico dentro do repositório GitHub
HISTORICO_PREFIX = "historico_atendimentos_"
HISTORICO_EXTENSION = ".parquet"

st.set_page_config(page_title="Dashboard Call Center", layout="wide")

# -------------------- Funções de Interação com a API do GitHub --------------------

def get_github_config():
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPO")
    branch = os.environ.get("GITHUB_BRANCH", "main")

    if token and repo:
        return token, repo, branch

    try:
        token  = st.secrets["github"]["token"]
        repo   = st.secrets["github"]["repo"]
        branch = st.secrets["github"].get("branch", "main")
        return token, repo, branch
    except KeyError:
        st.error("As credenciais do GitHub não estão configuradas.")
        return None, None, None
    except Exception as e:
        st.error(f"Erro ao carregar configurações do GitHub: {e}")
        return None, None, None

def get_github_headers():
    token, _, _ = get_github_config()
    if not token:
        return {}
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

def get_file_sha(path):
    token, repo, branch = get_github_config()
    if not token or not repo or not branch:
        return None
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    try:
        r   = requests.get(url, headers=get_github_headers())
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return data.get("sha")
        elif r.status_code == 404:
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão ao obter SHA: {e}")
    return None

def get_file_from_github(path):
    token, repo, branch = get_github_config()
    if not token or not repo or not branch:
        return None, None
    raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    try:
        r = requests.get(raw_url, headers={"Authorization": f"token {token}"})
        if r.status_code == 200 and len(r.content) > 0:
            return r.content, get_file_sha(path)
        elif r.status_code == 404:
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão ao baixar arquivo: {e}")
    return None, None

def save_file_to_github(path, content_bytes, message):
    token, repo, branch = get_github_config()
    if not token or not repo or not branch:
        return False
    sha = get_file_sha(path)
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch":  branch
    }
    if sha:
        payload["sha"] = sha
    try:
        r = requests.put(url, headers=get_github_headers(), data=json.dumps(payload))
        if r.status_code in [200, 201]:
            return True
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão ao salvar arquivo: {e}")
    return False

def delete_file_from_github(path, message):
    token, repo, branch = get_github_config()
    if not token or not repo or not branch:
        return False
    sha = get_file_sha(path)
    if not sha:
        return True
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    payload = {"message": message, "sha": sha, "branch": branch}
    try:
        r = requests.delete(url, headers=get_github_headers(), data=json.dumps(payload))
        if r.status_code == 200:
            return True
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão ao excluir arquivo: {e}")
    return False

def list_files_in_github_repo(path=""):
    token, repo, branch = get_github_config()
    if not token or not repo or not branch:
        return []
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    try:
        r = requests.get(url, headers=get_github_headers())
        if r.status_code == 200:
            return [item["path"] for item in r.json() if item["type"] == "file"]
        elif r.status_code == 404:
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão ao listar arquivos: {e}")
    return []

def df_to_parquet_bytes(df: pl.DataFrame):
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    return buf.getvalue()

def parquet_bytes_to_df(content_bytes):
    if not content_bytes:
        return pl.DataFrame()
    try:
        buf = io.BytesIO(content_bytes)
        buf.seek(0)
        return pl.read_parquet(buf)
    except Exception as e:
        st.error(f"Erro ao converter bytes Parquet para DataFrame: {e}")
        return pl.DataFrame()

# -------------------- Utils --------------------

def formatar_tempo(segundos):
    if segundos is None or np.isnan(segundos):
        return "-"
    segundos = int(segundos)
    h = segundos // 3600
    m = (segundos % 3600) // 60
    s = segundos % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def duracao_para_segundos(valor):
    if valor is None:
        return None
    s = str(valor).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.split(".")[0]
    partes = s.split(":")
    try:
        if len(partes) == 3:
            return float(int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2]))
        elif len(partes) == 2:
            return float(int(partes[0]) * 60 + int(partes[1]))
        else:
            return float(s)
    except Exception:
        return None

def normalizar_id(valor):
    if valor is None:
        return None
    s = str(valor).strip().lower()
    if not s or s == "nan":
        return None
    match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', s)
    return match.group(0) if match else None

def normalizar_col(nome):
    try:
        nome = nome.encode("latin-1").decode("utf-8")
    except Exception:
        pass
    nome = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return nome.strip().lower()

def _col_tma(df_cols):
    return "conversas_segundos" if "conversas_segundos" in df_cols else "duracao_segundos"

# -------------------- Mapa Genesys --------------------

MAPA_GENESYS = {
    "exportacao total concluida": "exportacao",
    "filtros":                    "filtros",
    "data":                       "data_atendimento_raw",
    "duracao":                    "duracao_str",
    "ani":                        "ani",
    "tipo de desconexao":         "tipo_desconexao",
    "total da ura":               "total_ura_str",
    "fila total":                 "fila_total_str",
    "total de conversas":         "total_conversas_str",
    "total de tpc":               "total_tpc_str",
    "tratamento total":           "tratamento_total_str",
    "tempo para abandonar":       "tempo_abandono_str",
    "id de conversa":             "id_genesys",
    "carimbo de data/hora do resultado parcial": "carimbo_parcial",
}

PADRAO_AGENTE = re.compile(r"usu.{0,15}interagiram", re.IGNORECASE)

def detectar_coluna_agente(colunas):
    for col in colunas:
        if PADRAO_AGENTE.search(normalizar_col(col)):
            return col
    return None

# -------------------- Carregamento --------------------

@st.cache_data(show_spinner="Carregando Genesys...", max_entries=3)
def carregar_genesys(file_bytes: bytes, file_name: str):
    try:
        df_raw = pl.read_excel(file_bytes, engine="calamine")

        renomear = {}
        for col in df_raw.columns:
            chave = normalizar_col(col)
            if chave in MAPA_GENESYS:
                renomear[col] = MAPA_GENESYS[chave]

        col_agente = detectar_coluna_agente(df_raw.columns)
        if col_agente:
            renomear[col_agente] = "nome_agente"

        df = df_raw.rename({k: v for k, v in renomear.items() if k in df_raw.columns})
        del df_raw
        gc.collect()

        if "exportacao" in df.columns:
            df = df.filter(pl.col("exportacao").cast(pl.Utf8).str.to_lowercase().str.strip_chars().is_in(["sim", "yes"]))

        if "filtros" in df.columns:
            df = df.with_columns(
                pl.col("filtros").cast(pl.Utf8).str.extract(r"Fila:\s*(.+)", 1).str.strip_chars().fill_null("URA_CORSAN").alias("fila")
            )
        elif "fila" not in df.columns:
            df = df.with_columns(pl.lit("URA_CORSAN").alias("fila"))

        if "data_atendimento_raw" in df.columns:
            df = df.with_columns(
                pl.col("data_atendimento_raw").cast(pl.Utf8).str.strip_chars().str.strptime(pl.Datetime, "%d/%m/%Y %H:%M", strict=False).alias("data_atendimento")
            )
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Datetime).alias("data_atendimento"))

        cols_tempo = {
            "duracao_str":          "duracao_segundos",
            "total_ura_str":        "ura_segundos",
            "fila_total_str":       "fila_segundos",
            "total_conversas_str":  "conversas_segundos",
            "total_tpc_str":        "tpc_segundos",
            "tratamento_total_str": "tratamento_segundos",
            "tempo_abandono_str":   "abandono_segundos",
        }

        for col_str, col_seg in cols_tempo.items():
            if col_str in df.columns:
                df = df.with_columns(
                    pl.col(col_str).map_elements(duracao_para_segundos, return_dtype=pl.Float64).alias(col_seg)
                )

        if "id_genesys" in df.columns:
            df = df.with_columns(
                pl.col("id_genesys").map_elements(normalizar_id, return_dtype=pl.Utf8).alias("id_genesys_norm")
            )
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("id_genesys_norm"))

        if "ani" in df.columns:
            df = df.with_columns(
                pl.col("ani").cast(pl.Utf8).str.replace(r"^tel:\+", "").str.strip_chars()
            )

        if "nome_agente" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("nome_agente").cast(pl.Utf8).str.to_lowercase().str.strip_chars().is_in(["nan", "", "none"]))
                .then(None)
                .otherwise(pl.col("nome_agente").cast(pl.Utf8).str.strip_chars())
                .alias("nome_agente")
            )

        st.info(f"Genesys: {df.height} interacoes carregadas.")
        return df

    except Exception as e:
        st.error(f"Erro ao carregar Genesys: {e}")
        return pl.DataFrame()


@st.cache_data(show_spinner="Carregando Zendesk...", max_entries=3)
def carregar_zendesk(file_bytes: bytes, file_name: str):
    try:
        df = pl.read_excel(file_bytes, engine="calamine")
        df = df.rename({col: col.strip() for col in df.columns})

        renomear = {
            "ID do ticket":                              "ticket_id",
            "Assuntos do Ticket":                        "assunto",
            "Criacao do ticket - Carimbo de data/hora":  "data_criacao_zen",
            "Criação do ticket - Carimbo de data/hora":  "data_criacao_zen",
            "ID Genesys":                                "id_genesys",
            "Matricula":                                 "matricula",
            "Tickets":                                   "tickets_zen",
        }
        df = df.rename({k: v for k, v in renomear.items() if k in df.columns})

        if "data_criacao_zen" in df.columns:
            df = df.with_columns(
                pl.col("data_criacao_zen").cast(pl.Utf8).str.strip_chars().str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False)
            )

        if "id_genesys" in df.columns:
            df = df.with_columns(
                pl.col("id_genesys").map_elements(normalizar_id, return_dtype=pl.Utf8).alias("id_genesys_norm")
            )

        total = df.height
        com_id = df.filter(pl.col("id_genesys_norm").is_not_null()).height if "id_genesys_norm" in df.columns else 0
        st.info(f"Zendesk: {total} tickets, {com_id} com ID Genesys.")
        return df

    except Exception as e:
        st.error(f"Erro ao carregar Zendesk: {e}")
        return pl.DataFrame()


# -------------------- Integracao --------------------

def integrar_dados(df_zen: pl.DataFrame, df_gen: pl.DataFrame):
    if df_gen.is_empty():
        st.error("Arquivo Genesys vazio apos processamento.")
        return pl.DataFrame()

    df = df_gen.clone()

    if (
        not df_zen.is_empty()
        and "id_genesys_norm" in df_zen.columns
        and "id_genesys_norm" in df.columns
        and df.filter(pl.col("id_genesys_norm").is_not_null()).height > 0
    ):
        colunas_zen = ["id_genesys_norm"]
        for col in ["ticket_id", "assunto", "matricula", "data_criacao_zen", "tickets_zen"]:
            if col in df_zen.columns:
                colunas_zen.append(col)

        df_zen_slim = df_zen.select(colunas_zen).unique(subset=["id_genesys_norm"])
        df = df.join(df_zen_slim, on="id_genesys_norm", how="left")

        total = df.height
        com_assunto = df.filter(pl.col("assunto").is_not_null()).height if "assunto" in df.columns else 0
        st.success(
            f"Merge concluido: {total} registros | "
            f"{com_assunto} cruzados com Zendesk ({com_assunto/total*100:.1f}%)"
        )
    else:
        if df_zen.is_empty():
            st.warning("Zendesk nao carregado; exibindo so dados do Genesys.")
        else:
            st.warning("ID de conversa nao disponivel para cruzamento.")
        df = df.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("ticket_id"),
            pl.lit(None).cast(pl.Utf8).alias("assunto"),
            pl.lit(None).cast(pl.Utf8).alias("matricula")
        ])

    df = df.with_columns(pl.col("data_atendimento").alias("data_base"))

    if "data_criacao_zen" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("data_base").is_null() & pl.col("data_criacao_zen").is_not_null())
            .then(pl.col("data_criacao_zen"))
            .otherwise(pl.col("data_base"))
            .alias("data_base")
        )

    if "data_base" in df.columns:
        df = df.with_columns(pl.col("data_base").dt.strftime("%Y-%m").alias("mes"))
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("mes"))

    return df


# -------------------- Historico --------------------

@st.cache_data(show_spinner="Carregando historico...", ttl=60)
def carregar_historico():
    all_files = list_files_in_github_repo()
    parquet_files = [f for f in all_files if f.startswith(HISTORICO_PREFIX) and f.endswith(HISTORICO_EXTENSION)]

    if not parquet_files:
        return pl.DataFrame()

    dfs = []
    for file_path in parquet_files:
        content_bytes, _ = get_file_from_github(file_path)
        if content_bytes:
            df_part = parquet_bytes_to_df(content_bytes)
            if not df_part.is_empty():
                dfs.append(df_part)

    if not dfs:
        return pl.DataFrame()

    try:
        df_final = pl.concat(dfs, how="diagonal_relaxed")
    except Exception as e:
        st.error(f"Erro ao juntar os arquivos de histórico: {e}")
        # Fallback de segurança caso o diagonal_relaxed falhe por versão do Polars
        df_final = pl.concat(dfs, how="diagonal")

    if "id_genesys_norm" in df_final.columns and df_final.filter(pl.col("id_genesys_norm").is_not_null()).height > 0:
        df_final = df_final.unique(subset=["id_genesys_norm"], keep="last")
    else:
        chaves = [c for c in ["nome_agente", "data_atendimento", "duracao_segundos"] if c in df_final.columns]
        if chaves:
            df_final = df_final.unique(subset=chaves, keep="last")

    return df_final

def salvar_novo_historico_parcial(df_novo_lote: pl.DataFrame):
    if df_novo_lote.is_empty():
        st.warning("Nenhum dado para salvar no novo arquivo de histórico.")
        return False

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_file_name = f"{HISTORICO_PREFIX}{timestamp}{HISTORICO_EXTENSION}"

    st.info(f"Tentando salvar novo arquivo de histórico no GitHub: '{new_file_name}'")
    content_bytes = df_to_parquet_bytes(df_novo_lote)

    if save_file_to_github(new_file_name, content_bytes, f"Adiciona novo lote de dados ({timestamp})"):
        carregar_historico.clear()
        return True
    return False

# -------------------- Filtros --------------------

def aplicar_filtros(df: pl.DataFrame):
    st.sidebar.header("Filtros")
    df_f = df.clone()

    if "data_base" in df_f.columns and df_f.filter(pl.col("data_base").is_not_null()).height > 0:
        min_data = df_f.select(pl.col("data_base").min()).item().date()
        max_data = df_f.select(pl.col("data_base").max()).item().date()

        periodo = st.sidebar.date_input(
            "Periodo",
            value=(min_data, max_data),
            min_value=min_data,
            max_value=max_data,
            key="filtro_periodo"
        )
        if isinstance(periodo, (list, tuple)) and len(periodo) == 2:
            ini = periodo[0]
            fim = periodo[1]
            df_f = df_f.filter((pl.col("data_base").dt.date() >= ini) & (pl.col("data_base").dt.date() <= fim))

    return df_f


# -------------------- Visao Geral --------------------

def secao_visao_geral(df: pl.DataFrame):
    st.subheader("Visao geral")

    col_tma = _col_tma(df.columns)

    total       = df.height
    tma_medio   = df.select(pl.col(col_tma).mean()).item() if col_tma in df.columns else np.nan
    dur_total   = df.select(pl.col("duracao_segundos").sum()).item() if "duracao_segundos" in df.columns else 0
    ura_medio   = df.select(pl.col("ura_segundos").mean()).item() if "ura_segundos" in df.columns else np.nan
    fila_medio  = df.select(pl.col("fila_segundos").mean()).item() if "fila_segundos" in df.columns else np.nan
    tpc_medio   = df.select(pl.col("tpc_segundos").mean()).item() if "tpc_segundos" in df.columns else np.nan
    trat_medio  = df.select(pl.col("tratamento_segundos").mean()).item() if "tratamento_segundos" in df.columns else np.nan
    aband_medio = df.select(pl.col("abandono_segundos").mean()).item() if "abandono_segundos" in df.columns else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total de atendimentos", total)
    m2.metric("TMA medio", formatar_tempo(tma_medio))
    m3.metric("Tempo total em atendimento", formatar_tempo(dur_total))
    m4.metric("Tempo medio na fila", formatar_tempo(fila_medio))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Tempo medio na URA", formatar_tempo(ura_medio))
    m6.metric("Tempo medio de conversa", formatar_tempo(tma_medio))
    m7.metric("Tempo medio de tratamento", formatar_tempo(trat_medio))
    m8.metric("Tempo medio ate abandono", formatar_tempo(aband_medio))

    st.markdown("---")

    if "data_base" in df.columns and df.filter(pl.col("data_base").is_not_null()).height > 0:
        df_dia = (
            df.filter(pl.col("data_base").is_not_null())
            .group_by(pl.col("data_base").dt.truncate("1d"))
            .len()
            .rename({"len": "atendimentos"})
            .sort("data_base")
        ).to_pandas()

        fig_dia = px.bar(
            df_dia, x="data_base", y="atendimentos", text="atendimentos",
            title="Atendimentos por dia",
            labels={"data_base": "Data", "atendimentos": "Atendimentos"}
        )
        fig_dia.update_traces(textposition="outside")
        fig_dia.update_xaxes(tickformat="%d/%m/%Y", dtick="86400000.0")
        st.plotly_chart(fig_dia, width="stretch", key="vg_dia")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        if "tipo_desconexao" in df.columns and df.filter(pl.col("tipo_desconexao").is_not_null()).height > 0:
            df_desc = (
                df.drop_nulls("tipo_desconexao")
                .group_by("tipo_desconexao")
                .len()
                .rename({"tipo_desconexao": "tipo", "len": "quantidade"})
            ).to_pandas()

            fig_desc = px.pie(
                df_desc, names="tipo", values="quantidade",
                title="Tipos de desconexao", hole=0.4
            )
            fig_desc.update_traces(textinfo="label+percent")
            st.plotly_chart(fig_desc, width="stretch", key="vg_desconexao")

    with c2:
        if "nome_agente" in df.columns and df.filter(pl.col("nome_agente").is_not_null()).height > 0:
            df_ag = (
                df.drop_nulls("nome_agente")
                .group_by("nome_agente")
                .len()
                .rename({"len": "atendimentos"})
                .sort("atendimentos", descending=True)
            ).to_pandas()

            fig_ag = px.bar(
                df_ag, x="nome_agente", y="atendimentos", text="atendimentos",
                title="Atendimentos por agente",
                labels={"nome_agente": "Agente", "atendimentos": "Atendimentos"}
            )
            fig_ag.update_traces(textposition="outside")
            fig_ag.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_ag, width="stretch", key="vg_agente")

    st.markdown("---")

    componentes = {
        "URA":          "ura_segundos",
        "Fila":         "fila_segundos",
        "Conversa":     "conversas_segundos",
        "TPC":          "tpc_segundos",
        "Tratamento":   "tratamento_segundos",
    }
    dados_comp = [
        {"componente": k, "media_s": df.select(pl.col(v).mean()).item()}
        for k, v in componentes.items()
        if v in df.columns and df.filter(pl.col(v).is_not_null()).height > 0
    ]
    if dados_comp:
        import pandas as pd
        df_comp = pd.DataFrame(dados_comp)
        df_comp["Tempo medio"] = df_comp["media_s"].apply(formatar_tempo)
        fig_comp = px.bar(
            df_comp, x="componente", y="media_s", text="Tempo medio",
            title="Tempo medio por componente (geral)",
            labels={"componente": "Componente", "media_s": "Segundos"}
        )
        fig_comp.update_traces(textposition="outside")
        st.plotly_chart(fig_comp, width="stretch", key="vg_componentes")

    st.markdown("---")

    if "assunto" in df.columns and df.filter(pl.col("assunto").is_not_null()).height > 0:
        df_ass = (
            df.drop_nulls("assunto")
            .group_by("assunto")
            .len()
            .rename({"len": "atendimentos"})
            .sort("atendimentos", descending=True)
            .head(15)
        ).to_pandas()

        fig_ass = px.bar(
            df_ass, x="assunto", y="atendimentos", text="atendimentos",
            title="Top 15 assuntos (volume)",
            labels={"assunto": "Assunto", "atendimentos": "Atendimentos"}
        )
        fig_ass.update_traces(textposition="outside")
        fig_ass.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_ass, width="stretch", key="vg_assunto")


# -------------------- Por Agente --------------------

def secao_por_agente(df: pl.DataFrame):
    st.subheader("Atendimentos por agente")

    if "nome_agente" not in df.columns or df.filter(pl.col("nome_agente").is_not_null()).height == 0:
        st.info("Sem dados de agente.")
        return

    col_tma = _col_tma(df.columns)

    df_ag = (
        df.drop_nulls("nome_agente")
        .group_by("nome_agente")
        .agg([
            pl.len().alias("atendimentos"),
            pl.col(col_tma).mean().alias("tma_s"),
            pl.col("duracao_segundos").sum().alias("tempo_total_s")
        ])
        .sort("atendimentos", descending=True)
    ).to_pandas()

    df_ag["TMA"]         = df_ag["tma_s"].apply(formatar_tempo)
    df_ag["Tempo Total"] = df_ag["tempo_total_s"].apply(formatar_tempo)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            df_ag, x="nome_agente", y="atendimentos", text="atendimentos",
            title="Atendimentos por agente",
            labels={"nome_agente": "Agente", "atendimentos": "Atendimentos"}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch", key="pa_atendimentos")
    with c2:
        fig2 = px.bar(
            df_ag, x="nome_agente", y="tma_s", text=df_ag["TMA"],
            title="TMA por agente",
            labels={"nome_agente": "Agente", "tma_s": "TMA (s)"}
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig2, width="stretch", key="pa_tma")

    st.dataframe(
        df_ag[["nome_agente", "atendimentos", "TMA", "Tempo Total"]],
        width="stretch"
    )


# -------------------- Detalhe Agente --------------------

def secao_detalhe_agente(df: pl.DataFrame):
    st.subheader("Detalhe por agente")

    if "nome_agente" not in df.columns or df.filter(pl.col("nome_agente").is_not_null()).height == 0:
        st.info("Sem dados de agente.")
        return

    agentes = sorted(df.drop_nulls("nome_agente").select("nome_agente").unique().to_series().to_list())
    agente_sel = st.selectbox("Selecione o agente", agentes, key="sel_agente_detalhe")

    df_ag = df.filter(pl.col("nome_agente") == agente_sel)
    if df_ag.is_empty():
        st.info("Sem dados para este agente.")
        return

    col_tma = _col_tma(df_ag.columns)

    total     = df_ag.height
    tma_med   = df_ag.select(pl.col(col_tma).mean()).item() if col_tma in df_ag.columns else np.nan
    dur_total = df_ag.select(pl.col("duracao_segundos").sum()).item() if "duracao_segundos" in df_ag.columns else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Atendimentos", total)
    m2.metric("TMA medio", formatar_tempo(tma_med))
    m3.metric("Tempo total", formatar_tempo(dur_total))

    st.markdown("---")

    componentes = {
        "URA":        "ura_segundos",
        "Fila":       "fila_segundos",
        "Conversa":   "conversas_segundos",
        "TPC":        "tpc_segundos",
        "Tratamento": "tratamento_segundos",
    }
    dados_comp = [
        {"componente": k, "media_s": df_ag.select(pl.col(v).mean()).item()}
        for k, v in componentes.items()
        if v in df_ag.columns and df_ag.filter(pl.col(v).is_not_null()).height > 0
    ]
    if dados_comp:
        import pandas as pd
        df_comp = pd.DataFrame(dados_comp)
        df_comp["Tempo medio"] = df_comp["media_s"].apply(formatar_tempo)
        fig = px.bar(
            df_comp, x="componente", y="media_s", text="Tempo medio",
            title=f"Tempo medio por componente - {agente_sel}",
            labels={"componente": "Componente", "media_s": "Segundos"}
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch", key="da_componentes")

    st.markdown("---")

    if "tipo_desconexao" in df_ag.columns and df_ag.filter(pl.col("tipo_desconexao").is_not_null()).height > 0:
        df_desc = (
            df_ag.drop_nulls("tipo_desconexao")
            .group_by("tipo_desconexao")
            .len()
            .rename({"tipo_desconexao": "tipo", "len": "quantidade"})
        ).to_pandas()

        df_desc["pct"] = (df_desc["quantidade"] / df_desc["quantidade"].sum() * 100).round(1)

        c1, c2 = st.columns(2)
        with c1:
            fig_d = px.pie(
                df_desc, names="tipo", values="quantidade",
                title="Tipos de desconexao", hole=0.4
            )
            fig_d.update_traces(textinfo="label+percent")
            st.plotly_chart(fig_d, width="stretch", key="da_desconexao_pie")
        with c2:
            st.dataframe(
                df_desc.rename(columns={"tipo": "Tipo", "quantidade": "Qtd", "pct": "%"}),
                width="stretch"
            )

    st.markdown("---")

    if "data_base" in df_ag.columns and df_ag.filter(pl.col("data_base").is_not_null()).height > 0:
        df_dia = (
            df_ag.filter(pl.col("data_base").is_not_null())
            .group_by(pl.col("data_base").dt.truncate("1d"))
            .len()
            .rename({"len": "atendimentos"})
            .sort("data_base")
        ).to_pandas()

        fig2 = px.bar(
            df_dia, x="data_base", y="atendimentos", text="atendimentos",
            title=f"Volume diario - {agente_sel}",
            labels={"data_base": "Data", "atendimentos": "Atendimentos"}
        )
        fig2.update_traces(textposition="outside")
        fig2.update_xaxes(tickformat="%d/%m/%Y", dtick="86400000.0")
        st.plotly_chart(fig2, width="stretch", key="da_volume_diario")


# -------------------- Por Assunto --------------------

def secao_por_assunto(df: pl.DataFrame):
    st.subheader("Atendimentos por assunto")

    if "assunto" not in df.columns or df.filter(pl.col("assunto").is_not_null()).height == 0:
        st.info("Ainda nao ha assuntos cruzados com o Zendesk.")
        return

    col_tma = _col_tma(df.columns)

    df_ass = (
        df.drop_nulls("assunto")
        .group_by("assunto")
        .agg([
            pl.len().alias("atendimentos"),
            pl.col(col_tma).mean().alias("tma_s"),
            pl.col("duracao_segundos").sum().alias("tempo_total_s")
        ])
        .sort("atendimentos", descending=True)
    ).to_pandas()

    df_ass["TMA"]         = df_ass["tma_s"].apply(formatar_tempo)
    df_ass["Tempo Total"] = df_ass["tempo_total_s"].apply(formatar_tempo)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            df_ass, x="assunto", y="atendimentos", text="atendimentos",
            title="Volume por assunto",
            labels={"assunto": "Assunto", "atendimentos": "Atendimentos"}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch", key="ass_volume")
    with c2:
        fig2 = px.bar(
            df_ass, x="assunto", y="tma_s", text=df_ass["TMA"],
            title="TMA por assunto",
            labels={"assunto": "Assunto", "tma_s": "TMA (s)"}
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig2, width="stretch", key="ass_tma")

    st.dataframe(
        df_ass[["assunto", "atendimentos", "TMA", "Tempo Total"]],
        width="stretch"
    )


# -------------------- Top TMA por mes --------------------

def secao_top_assuntos_tma(df: pl.DataFrame):
    st.subheader("Top 10 assuntos por TMA - por mes")

    if "assunto" not in df.columns or df.filter(pl.col("assunto").is_not_null()).height == 0:
        st.info("Ainda nao ha assuntos cruzados com o Zendesk.")
        return

    if "mes" not in df.columns or df.filter(pl.col("mes").is_not_null()).height == 0:
        st.info("Coluna de mes nao disponivel.")
        return

    col_tma = _col_tma(df.columns)
    meses   = sorted(df.drop_nulls("mes").select("mes").unique().to_series().to_list())
    mes_sel = st.selectbox("Selecione o mes", meses, key="sel_mes_top_tma")

    df_mes = df.filter((pl.col("mes") == mes_sel) & pl.col("assunto").is_not_null())
    if df_mes.is_empty():
        st.info("Sem dados para este mes.")
        return

    df_top = (
        df_mes.group_by("assunto")
        .agg([
            pl.len().alias("atendimentos"),
            pl.col(col_tma).mean().alias("tma_s")
        ])
        .sort("tma_s", descending=True)
        .head(10)
    ).to_pandas()

    df_top["TMA"] = df_top["tma_s"].apply(formatar_tempo)

    fig = px.bar(
        df_top.sort_values("tma_s", ascending=True),
        x="tma_s", y="assunto", orientation="h",
        text="TMA", color="tma_s", color_continuous_scale="Reds",
        title=f"Top 10 assuntos com maior TMA - {mes_sel}",
        labels={"tma_s": "TMA (s)", "assunto": "Assunto"}
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch", key="top_tma_bar")

    st.dataframe(
        df_top[["assunto", "atendimentos", "TMA"]].reset_index(drop=True),
        width="stretch"
    )

    if len(meses) > 1:
        st.markdown("**Comparativo entre meses**")
        df_comp = (
            df.drop_nulls("assunto")
            .group_by(["mes", "assunto"])
            .agg(pl.col(col_tma).mean().alias("tma_s"))
        ).to_pandas()

        tops = []
        for m in meses:
            bloco = (
                df_comp[df_comp["mes"] == m]
                .sort_values("tma_s", ascending=False)
                .head(10)
            )
            tops.append(bloco)

        import pandas as pd
        df_comp_final = pd.concat(tops, ignore_index=True)
        df_comp_final["TMA"] = df_comp_final["tma_s"].apply(formatar_tempo)

        fig2 = px.bar(
            df_comp_final, x="assunto", y="tma_s", color="mes",
            barmode="group", text="TMA",
            title="TMA por assunto - comparativo entre meses",
            labels={"tma_s": "TMA (s)", "assunto": "Assunto", "mes": "Mes"}
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig2, width="stretch", key="top_tma_comp")


# -------------------- Upload & main --------------------

def secao_upload():
    st.sidebar.header("Upload mensal")

    arq_zen = st.sidebar.file_uploader("Zendesk (XLSX)", type=["xlsx", "xls"])
    arq_gen = st.sidebar.file_uploader("Genesys (XLSX)", type=["xlsx", "xls"])

    if arq_gen is not None:
        if st.sidebar.button("Processar e acumular"):
            df_zen = carregar_zendesk(arq_zen.read(), arq_zen.name) if arq_zen else pl.DataFrame()
            df_gen = carregar_genesys(arq_gen.read(), arq_gen.name)
            df_novo = integrar_dados(df_zen, df_gen)

            if df_novo.is_empty():
                st.sidebar.error("Nenhum dado gerado.")
                return

            if salvar_novo_historico_parcial(df_novo):
                st.sidebar.success(f"Novo lote de dados salvo no GitHub. Total de {df_novo.height} registros.")
                st.rerun()
            else:
                st.sidebar.error("Falha ao salvar o novo lote de dados no GitHub.")

    with st.sidebar.expander("Gerenciar historico"):
        st.warning("Esta seção interage diretamente com o repositório GitHub.")

        if st.button("Listar arquivos de histórico"):
            parquet_files = [f for f in list_files_in_github_repo() if f.startswith(HISTORICO_PREFIX) and f.endswith(HISTORICO_EXTENSION)]
            if parquet_files:
                st.write("Arquivos de histórico no GitHub:")
                for f in parquet_files:
                    st.write(f"- {f}")
            else:
                st.info("Nenhum arquivo de histórico encontrado no GitHub.")

        confirm = st.checkbox("Confirmar exclusao de TODOS os arquivos de historico?")
        if confirm:
            if st.button("Apagar TODOS os arquivos de histórico do GitHub", type="primary"):
                parquet_files = [f for f in list_files_in_github_repo() if f.startswith(HISTORICO_PREFIX) and f.endswith(HISTORICO_EXTENSION)]
                if not parquet_files:
                    st.info("Nenhum arquivo de histórico para apagar.")
                else:
                    st.info(f"Apagando {len(parquet_files)} arquivos de histórico...")
                    all_deleted = True
                    for file_path in parquet_files:
                        if not delete_file_from_github(file_path, f"Exclui arquivo de histórico '{file_path}' via Streamlit"):
                            all_deleted = False
                            st.error(f"Falha ao apagar '{file_path}'.")
                    if all_deleted:
                        carregar_historico.clear()
                        st.success("Todos os arquivos de histórico foram apagados do GitHub.")
                        st.rerun()
                    else:
                        st.error("Alguns arquivos de histórico não puderam ser apagados.")


# -------------------- Autenticação --------------------

def get_users():
    users = {}
    try:
        # Verifica se a seção [users] existe
        if "users" not in st.secrets:
            st.error("🚨 ERRO: A seção '[users]' não foi encontrada. Verifique a variável STREAMLIT_SECRETS.")
            return users

        secrets  = st.secrets["users"]
        prefixes = set()
        for key in secrets:
            if key.endswith("_user"):
                prefixes.add(key[:-5])

        for prefix in prefixes:
            username = secrets.get(f"{prefix}_user", "")
            password = secrets.get(f"{prefix}_password", "")
            role     = secrets.get(f"{prefix}_role", "user")
            if username:
                users[username] = {"password": password, "role": role}

        if not users:
            st.warning("🚨 A seção '[users]' existe, mas nenhum usuário foi carregado.")

    except Exception as e:
        st.error(f"🚨 Erro interno ao ler usuários: {e}")

    return users

def login_screen():
    st.title("🔐 Login")
    st.markdown("Faça login para acessar o sistema.")
    with st.form("login_form"):
        username  = st.text_input("Usuário")
        password  = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar")
    if submitted:
        users = get_users()
        if username in users and str(users[username]["password"]) == str(password):
            st.session_state["logged_in"] = True
            st.session_state["username"]  = username
            st.session_state["role"]      = users[username]["role"]
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos.")

def is_admin():
    return st.session_state.get("role") == "admin"

def logout():
    st.session_state.clear()
    st.rerun()

def main():
    # Trava o aplicativo na tela de login se não estiver autenticado
    if not st.session_state.get("logged_in", False):
        login_screen()
        return

    # Cabeçalho da barra lateral com botão de sair
    st.sidebar.markdown(f"👤 Logado como: **{st.session_state.get('username')}**")
    if st.sidebar.button("Sair / Logout"):
        logout()
    st.sidebar.markdown("---")

    st.title("Dashboard de Atendimentos - Call Center")

    # Carrega o histórico completo na inicialização
    df_hist = carregar_historico()

    # Controle de Acesso: Apenas ADMIN vê a seção de upload e exclusão
    if is_admin():
        secao_upload()
    else:
        st.sidebar.info("Modo de visualização.")

    # Verifica se o DataFrame do Polars está vazio usando .is_empty()
    if df_hist.is_empty():
        st.info("Faça o upload do arquivo Genesys (XLSX) para começar, ou verifique se há arquivos de histórico no GitHub e as credenciais estão corretas.")
        return

    df_filtrado = aplicar_filtros(df_hist)

    if df_filtrado.is_empty():
        st.warning("Nenhum registro para os filtros atuais.")
        return

    aba1, aba2, aba3, aba4, aba5 = st.tabs([
        "Visao geral",
        "Por agente",
        "Detalhe do agente",
        "Por assunto",
        "Top TMA por mes",
    ])
    with aba1: secao_visao_geral(df_filtrado)
    with aba2: secao_por_agente(df_filtrado)
    with aba3: secao_detalhe_agente(df_filtrado)
    with aba4: secao_por_assunto(df_filtrado)
    with aba5: secao_top_assuntos_tma(df_filtrado)


if __name__ == "__main__":
    main()

