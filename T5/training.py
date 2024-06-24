from transformers import TrainingArguments, Trainer

def train_model(model, train_dataset, valid_dataset, output_dir, epochs, batch_size, learning_rate):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=output_dir,
        logging_steps=10,
        evaluation_strategy='steps',
        save_steps=500,
        eval_steps=500,
        load_best_model_at_end=True,
        save_total_limit=5,
        report_to='tensorboard',
        learning_rate=learning_rate,
        fp16=True,
        dataloader_num_workers=4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()
    return trainer
