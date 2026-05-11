TARGETS_MAP = {'at': 'Atheism',
             'cc':'Climate Change is a Real Concern',
             'fm':'Feminist Movement',
             'hc':'Hillary Clinton',
             'la':'Legalization of Abortion',
             'dt':'Donald Trump',
            'dtp':'Donald Trump',
             'bernie':'Bernie Sanders',
             'joe':'Joe Biden',
            'face':'face_masks',
            'fauci': 'fauci',
            'school':'school_closures',
            'stay':'stay_at_home_orders',     
            'ent':'entertainment',
            'hlt':'healthcare'
              }
SEMEVAL_LABELS = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}
WTWT_LABELS = {'COMMENT': 0, 'REFUTE': 1, 'SUPPORT': 2, 'UNRELATED': 3}

KNOWLEDGE_BASE = {
    'at': "Atheism is the disbelief or lack of belief in the existence of God or gods.",
    'cc': "Climate Change refers to significant and lasting changes in global weather patterns, often attributed to human activities leading to rising temperatures and extreme weather events.",
    'fm': "The Feminist Movement is a series of social and political campaigns for reforms on issues such as reproductive rights, domestic violence, equal pay, and gender equality.",
    'hc': "Hillary Clinton is an American politician, former First Lady, Senator from New York, and Secretary of State, a prominent figure in the Democratic Party.",
    'la': "Legalization of Abortion refers to the legal right for individuals to terminate a pregnancy, a highly debated topic concerning reproductive rights and ethical considerations.",
    'dt': "Donald Trump is an American businessman, television personality, and politician who served as the 45th president of the United States, a prominent figure in the Republican Party.",
    'dtp': "Donald Trump is an American businessman, television personality, and politician who served as the 45th president of the United States, a prominent figure in the Republican Party.",
    "ent": "Tweets concern the entertainment-sector M&A in which The Walt Disney Company acquired 21st Century Fox (DIS→FOXA). Stance labels indicate whether the tweet asserts the deal will/has gone through (Support), will not/was blocked (Refute), comments on implications without taking a position (Comment), or is unrelated.",
    "hlt": "Tweets concern four healthcare-sector M&A operations among U.S. insurers/pharmacy firms: CVS↔Aetna and Cigna↔Express Scripts (both succeeded), Anthem↔Cigna and Aetna↔Humana (both blocked). Stance labels indicate whether the tweet asserts the respective deal will/has gone through (Support), will not/was blocked (Refute), comments without a position (Comment), or is unrelated.",
    'pol': "Politics refers to the activities associated with the governance of a country or area, especially the debate or conflict among individuals or parties having or hoping to achieve power.",
    'bernie': "Bernie Sanders is an American politician who has served as the junior United States senator from Vermont since 2007, a prominent figure in progressive politics.",
    'joe': "Joe Biden is an American politician who is the 46th and current president of the United States, a prominent figure in the Democratic Party.",
    'face': "Face masks are protective coverings worn over the mouth and nose, often used to prevent the spread of respiratory diseases, particularly during pandemics.",
    'fauci': "Anthony Fauci is an American physician-scientist and immunologist who served as the director of the National Institute of Allergy and Infectious Diseases (NIAID) and chief medical advisor to the president, prominent during the COVID-19 pandemic.",
    'school': "School closures refer to the temporary or permanent shutdown of educational institutions, often in response to public health crises or natural disasters.",
    'stay': "Stay-at-home orders are government-mandated directives requiring citizens to remain in their residences, typically enacted during public health emergencies to limit disease spread."
}
TARGET_DATASET_MAP = {
    'at': 'semeval',
    'cc': 'semeval',
    'fm': 'semeval',
    'hc': 'semeval',
    'la': 'semeval',
    'dt': 'semeval',
    'dtp': 'pstance', 
    'bernie': 'pstance',
    'joe': 'pstance',
    'face': 'covid',
    'fauci': 'covid',
    'school': 'covid',
    'stay': 'covid',
    'ent': 'wtwt', 
    'hlt': 'wtwt', 
}

TARGET_MODULES_MAP = {
    "llama3_8b": ["q_proj", "v_proj"],
    "mistral_7b": ["q_proj", "v_proj"],
    "mistral_24b": ["q_proj", "v_proj"],
    "phi": [ "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj"]  # For Phi-3
}


