cd ..\llama\llama-b11554-bin-win-vulkan-x64
.\llama-server.exe -m ..\..\Model\Qwen2.5-7B-Instruct-Q4_K_M.gguf -ngl 99 -c 4096 -np 1 --log-file server_log.txt
pause