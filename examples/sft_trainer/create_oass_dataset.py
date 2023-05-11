from datasets import load_dataset, Dataset


ds = load_dataset("OpenAssistant/oasst1", split="train+validation[:1%]", revision="fd7dd3fc13915d6a3bba7e4e854b940b651a4032")
df = ds.to_pandas()


def build_conversations(df):
    conversations = []
    roots = set(df["message_id"]) - set(df["parent_id"])
    for root in roots:
        conv = []
        curr_id = root
        while curr_id is not None:
            message = df[df["message_id"] == curr_id].iloc[0]
            role = message["role"]
            if role == "prompter":
                role = "user"
            conv.append({"role": role, "content": message["text"]})
            curr_id = message["parent_id"] if message["parent_id"] != "" else None
        conversations.append(conv[::-1])
    return conversations


trees = df.groupby("message_tree_id", sort=False).apply(build_conversations)
assert len(trees) == df["message_tree_id"].nunique()


messages = [msg for tree in trees for msg in tree]

def post_process(message):
    final_msg = ""
    for dict_elem in message:
        final_msg += "### " + dict_elem["role"] + ": " + dict_elem["content"] + "\n"
    return final_msg


msg_ds = Dataset.from_dict({"messages": [post_process(message) for message in messages]})
msg_ds = msg_ds.train_test_split(test_size=0.1, seed=42)


msg_ds["train"].push_to_hub("ybelkada/oasst1-tiny-subset", split="train")
msg_ds["test"].push_to_hub("ybelkada/oasst1-tiny-subset", split="test")