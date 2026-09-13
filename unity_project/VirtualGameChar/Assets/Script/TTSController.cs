using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;

public class TTSController : MonoBehaviour
{
    // --- TTS plugin ---
    [DllImport("tts_plugin", CharSet = CharSet.Unicode)]
    private static extern bool InitTTS(string modelPath);

    [DllImport("tts_plugin")]
    private static extern int Synthesize(long[] phonemeIds, int numPhonemes, float[] outputBuffer, int maxOutputSamples);

    [DllImport("tts_plugin")]
    private static extern void ShutdownTTS();

    public TextMeshProUGUI statsText;

    private AudioSource audioSource;
    private bool initialized = false;      // TTS
    private bool llmInitialized = false;   // llm_cli.exe present
    private const int SampleRate = 22050;

    private string llmCliPath;

    private static readonly long[] HelloPhonemes = {1, 0, 20, 0, 59, 0, 24, 0, 120, 0, 27, 0, 100, 0, 3, 0, 41, 0, 120, 0, 61, 0, 88, 0, 10, 0, 2};

    void Start()
    {
        audioSource = GetComponent<AudioSource>();

        string ttsModelPath = System.IO.Path.Combine(Application.streamingAssetsPath, "en_US-lessac-medium.onnx");
        initialized = InitTTS(ttsModelPath);
        Debug.Log("TTS Initialized: " + initialized);

        llmCliPath = "C:\\VirtualCharEngine\\cpp_wrapper\\llm_cli.exe";
        llmInitialized = System.IO.File.Exists(llmCliPath);
        Debug.Log("LLM CLI found: " + llmInitialized + " at " + llmCliPath);

        if (statsText != null)
            statsText.text = (initialized && llmInitialized) ? "Ready." : "Initialization failed — check Console.";
    }

    // Fixed-phrase TTS only (original button)
    public async void SpeakHello()
    {
        if (!initialized)
        {
            Debug.LogError("TTS not initialized");
            return;
        }

        float[] buffer = new float[100000];
        long[] phonemes = HelloPhonemes;

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        int sampleCount = await Task.Run(() => Synthesize(phonemes, phonemes.Length, buffer, buffer.Length));
        stopwatch.Stop();

        if (sampleCount > 0)
        {
            float[] audioData = new float[sampleCount];
            Array.Copy(buffer, audioData, sampleCount);
            PlayAudio(audioData);

            float synthesisMs = stopwatch.ElapsedMilliseconds;
            float audioDurationMs = (sampleCount / (float)SampleRate) * 1000f;
            float rtf = synthesisMs / audioDurationMs;

            if (statsText != null)
                statsText.text = $"Synthesis: {synthesisMs:F1} ms | Audio: {audioDurationMs:F1} ms | RTF: {rtf:F3}";
        }
        else
        {
            Debug.LogError("Synthesis failed");
            if (statsText != null) statsText.text = "Synthesis failed.";
        }
    }

    private const string SystemPrompt =
        "<|im_start|>system\nYou are a friendly shopkeeper character in a video game. " +
        "Keep every reply to one short sentence.<|im_end|>\n";

    // First entry is the opening line the NPC speaks unprompted; the rest are scripted "player" questions.
    private static readonly string[] ScriptedPlayerLines =
    {
        "Greet the player and ask how you can help them today.",
        "What items do you have for sale?",
        "Can you tell me about this place?"
    };

    // Full chain: LLM generates (with running conversation history) -> Python bridge phonemizes -> TTS speaks
    public async void ThinkAndSpeak()
    {
        if (!llmInitialized || !initialized)
        {
            Debug.LogError("LLM or TTS not initialized");
            return;
        }

        string history = SystemPrompt;
        string transcript = "";

        for (int turn = 0; turn < ScriptedPlayerLines.Length; turn++)
        {
            string userLine = ScriptedPlayerLines[turn];
            history += $"<|im_start|>user\n{userLine}<|im_end|>\n<|im_start|>assistant\n";

            var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();

            var llmStopwatch = System.Diagnostics.Stopwatch.StartNew();
            string generatedText = await Task.Run(() => GenerateTextViaProcess(history));
            llmStopwatch.Stop();

            if (string.IsNullOrEmpty(generatedText))
            {
                Debug.LogError($"LLM generation failed on turn {turn}");
                return;
            }
            generatedText = generatedText.Trim();
            Debug.Log($"Turn {turn} generated: {generatedText}");

            history += generatedText + "<|im_end|>\n";
            transcript += (turn > 0 ? $"Player: {userLine}\n" : "") + $"NPC: {generatedText}\n";
            if (statsText != null) statsText.text = transcript;

            var phonemizeStopwatch = System.Diagnostics.Stopwatch.StartNew();
            long[] phonemeIds = await Task.Run(() => PhonemizeText(generatedText));
            phonemizeStopwatch.Stop();

            if (phonemeIds == null || phonemeIds.Length == 0)
            {
                Debug.LogError($"Phonemization failed on turn {turn}");
                continue;
            }

            float[] buffer2 = new float[200000];
            var synthStopwatch = System.Diagnostics.Stopwatch.StartNew();
            int sampleCount = await Task.Run(() => Synthesize(phonemeIds, phonemeIds.Length, buffer2, buffer2.Length));
            synthStopwatch.Stop();
            totalStopwatch.Stop();

            if (sampleCount > 0)
            {
                float[] audioData = new float[sampleCount];
                Array.Copy(buffer2, audioData, sampleCount);
                PlayAudio(audioData);

                string timing = $"LLM: {llmStopwatch.ElapsedMilliseconds} ms | Phonemize: {phonemizeStopwatch.ElapsedMilliseconds} ms | " +
                                 $"Synthesis: {synthStopwatch.ElapsedMilliseconds} ms | Total: {totalStopwatch.ElapsedMilliseconds} ms";
                Debug.Log($"Turn {turn} timing — {timing}");

                float audioDurationSec = sampleCount / (float)SampleRate;
                await Task.Delay(Mathf.CeilToInt(audioDurationSec * 1000) + 400);
            }
            else
            {
                Debug.LogError($"Synthesis failed on turn {turn}");
            }
        }
    }

    private string GenerateTextViaProcess(string prompt)
    {
        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = llmCliPath,
            WorkingDirectory = "C:\\VirtualCharEngine\\cpp_wrapper",
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        using (var process = System.Diagnostics.Process.Start(psi))
        {
            var stdoutTask = process.StandardOutput.ReadToEndAsync();
            var stderrTask = process.StandardError.ReadToEndAsync();
            process.StandardInput.Write(prompt);
            process.StandardInput.Close();
            process.WaitForExit();
            string output = stdoutTask.Result.Trim();
            string error = stderrTask.Result;
            if (!string.IsNullOrEmpty(error)) Debug.Log("llm_cli stderr (diagnostic log): " + error.Length + " chars");
            return string.IsNullOrEmpty(output) ? null : output;
        }
    }

    private long[] PhonemizeText(string text)
    {
        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = "C:\\VirtualCharEngine\\venv\\Scripts\\python.exe",
            Arguments = "C:\\VirtualCharEngine\\phonemize_bridge.py",
            WorkingDirectory = "C:\\VirtualCharEngine",
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        using (var process = System.Diagnostics.Process.Start(psi))
        {
            var stdoutTask = process.StandardOutput.ReadToEndAsync();
            var stderrTask = process.StandardError.ReadToEndAsync();
            process.StandardInput.Write(text);
            process.StandardInput.Close();
            process.WaitForExit();
            string output = stdoutTask.Result.Trim();
            string error = stderrTask.Result;
            if (!string.IsNullOrEmpty(error)) Debug.LogWarning("phonemize_bridge stderr: " + error);
            if (string.IsNullOrEmpty(output)) return null;
            return output.Split(',').Select(long.Parse).ToArray();
        }
    }

    void PlayAudio(float[] samples)
    {
        AudioClip clip = AudioClip.Create("TTSOutput", samples.Length, 1, SampleRate, false);
        clip.SetData(samples, 0);
        audioSource.clip = clip;
        audioSource.Play();
    }

    void OnDestroy()
    {
        if (initialized) ShutdownTTS();
    }
}