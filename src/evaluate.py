from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, train_input, train_target, test_input, test_target):
    print("*** 훈련 세트 성능 평가 ***")
    train_pred = model.predict(train_input)
    
    # 정오 행렬(Confusion Matrix) 출력
    train_conf_matrix = confusion_matrix(train_target, train_pred)
    print("Confusion Matrix:")
    print("[[TN, FP], [FN, TP]")
    print(train_conf_matrix.tolist(), "\n")
    
    # 분류 리포트 출력 (정밀도, 재현율, F1-score 등)
    train_report = classification_report(train_target, train_pred, target_names=['양호(0)', '나쁨(1)'], zero_division=0)
    print("Classification Report:")
    print(train_report)
    print("-" * 50)

    # --- 테스트 세트 성능 평가 ---
    print("\n*** 테스트 세트 성능 평가 ***")
    test_pred = model.predict(test_input)
    
    # 정오 행렬(Confusion Matrix) 출력
    test_conf_matrix = confusion_matrix(test_target, test_pred)
    print("Confusion Matrix:")
    print("[[TN, FP], [FN, TP]")
    print(test_conf_matrix.tolist(), "\n")

    # 분류 리포트 출력
    test_report = classification_report(test_target, test_pred, target_names=['양호(0)', '나쁨(1)'], zero_division=0)
    print("Classification Report:")
    print(test_report)