from datasets import get_dataset_config_names

xtreme_subsets = get_dataset_config_names("xtreme")
print(f"XTREME has {len(xtreme_subsets)} configurations")

panx_subsets = [s for s in xtreme_subsets if s.startswith("PAN")]
print(panx_subsets[:3])