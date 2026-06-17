import torch


INTENT_INDEX = {
    "EXPANSION": 0,
    "COST_PRESSURE": 1,
    "STRATEGIC_PROBING": 2,
    "GENERAL_UPDATE": 3
}


def vectorize_conversation(intents):

    vec = torch.zeros(3, len(INTENT_INDEX))

    speaker_map = {
        "CEO": 0,
        "CFO": 1,
        "ANALYST": 2
    }

    for item in intents:

        speaker = item["speaker"]
        intent = item["intent"]

        if speaker in speaker_map and intent in INTENT_INDEX:

            vec[speaker_map[speaker]][INTENT_INDEX[intent]] = 1

    return vec.flatten()