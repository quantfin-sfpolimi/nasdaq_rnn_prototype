from hashlib import sha256

# The function aim to create a X_train array but hashed, and a clean X_test array, then return both.
def hashTrainTest(X_train, X_test, hashed_train_data=None):

  # Train test hashing
  
  hashed_train_data = []
  for x in X_train:
    x = str(x)
    hashed_x = sha256(x.encode('utf-8')).hexdigest()
    hashed_train_data.append(hashed_x)

  i = 0
  while i < len(X_test):
    x = str(X_test[i])
    hashed_x = sha256(x.encode('utf-8')).hexdigest()
    if hashed_x in hashed_train_data:
      X_test.pop(i)
    else:
      i+=1

  return (hashed_train_data, X_test)