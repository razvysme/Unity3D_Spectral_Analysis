using UnityEngine;
public class MovingAverageFilter
{
    private readonly float[] buffer;
    private int bufferIndex;
    private float sum;
    private readonly int bufferSize;

    public MovingAverageFilter(int size)
    {
        buffer = new float[size];
        bufferIndex = 0;
        sum = 0f;
        bufferSize = size;
    }

    public float AddValue(float newValue)
    {
        //Debug.Log($"Filter Input = {newValue}");
        // Subtract the oldest value from the sum
        sum -= buffer[bufferIndex];
        // Add the new value to the buffer and the sum
        buffer[bufferIndex] = newValue;
        sum += newValue;

        // Increment the buffer index and wrap around if necessary
        bufferIndex = (bufferIndex + 1) % bufferSize;

        //Debug.Log($"Filter Output = {sum / bufferSize}");
        // Return the average
        return sum / bufferSize;
    }
}
