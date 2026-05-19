# src/graph_builder.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import networkx as nx
from database.db_connections import get_engine

def build_partnership_graph(min_pairs=5):
    """Build a graph of batter pair partnerships: players who batted together in same innings of match."""
    engine = get_engine()
    q = """
    SELECT season, match_key, batting_team, inning, batter, non_striker
    FROM matches
    WHERE batter IS NOT NULL
    """
    df = pd.read_sql(q, engine)
    # Build edges: for each match_key & inning, collect unique batters and link all combinations
    G = nx.Graph()
    grouped = df.groupby(['match_key', 'inning'])
    for (mk, inn), sub in grouped:
        players = sub['batter'].unique().tolist()
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                a, b = players[i], players[j]
                if G.has_edge(a,b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a,b, weight=1)
    # filter edges under min_pairs
    to_remove = [(u,v) for u,v,d in G.edges(data=True) if d['weight'] < min_pairs]
    G.remove_edges_from(to_remove)
    return G
