# Debugging
## Solution to the debugging tasks from NHL as a part of Computer Vision & Data Science intake.

link to the code: https://github.com/heesos/debugging_nhl/blob/main/debugging.ipynb

## Task 1
The solution here was to change the fruits parameter to List type making all of the items in it ordered
```python
def id_to_fruit(fruit_id: int, fruits: List[str]) -> str:
```

## Task 2

First obvious bug was that there was a wrong column put in the swap line.
I have changed the column 1 to 0
```python
coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3] = coords[:, 1], coords[:, 0], coords[:,3],coords[:, 2]
```

This was not enough, because we kept assigning only Y value to the X. It did not work vice versa
The solution to it was making the hard copy of the coordinates and fetching the data from the copy array.
```python
copy=coords.copy()
coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3] = coords[:, 1], copy[:, 0], coords[:, 3], copy[:, 2]
```


## Task 3
Values from csv were intially fetched as strings had to change them to floats
```python
coords_array = np.array(results, dtype=float)
```

## Task 4

The issue with 64 batches was because there are 60 000 MNIST pictures. 60 000/64 = 937.5
At the end we were left with 32 pictures rather than 64.
real_samples were 32 while generated_samples were still 64 which gave 96 in total.
The solution to this bug was to reassign the value of batch_size in for loop.
```python
batch_size=real_samples.size()[0]
```

The visual bug: I believe it was the issue that epoch number was showing incorrectly in the plot.
```python
 name = f"Generate images\n Epoch: {epoch + 1} Loss D.: {loss_discriminator:.2f} Loss G.: {loss_generator:.2f}"
```
