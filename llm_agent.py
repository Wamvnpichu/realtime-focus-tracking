import os
import threading
import queue

class FocusLLMAgent:
    def __init__(self, api_key="YOUR_API_KEY_HERE"):
        self.api_key = api_key
        self.client = None
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except ImportError:
                print("[WARNING] google-genai package not found. Run: pip install google-genai")
                
        self.queue = queue.Queue()
        self.current_message = "Waiting for data..."
        self.is_running = True
        self.last_call_time = 0.0
        
        # Start worker thread
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def update_context(self, action_name, metrics):
        """Queue a new context for the LLM to generate a message."""
        import time
        current_time = time.time()
        
        # Strict 15-second cooldown to prevent Gemini API rate limit errors (429)
        if current_time - self.last_call_time < 15.0:
            return
            
        # Avoid overwhelming the queue
        if self.queue.qsize() < 2:
            self.queue.put((action_name, metrics))
            self.last_call_time = current_time
            
    def get_current_message(self):
        """Retrieve the latest generated message."""
        return self.current_message
        
    def reset_message(self):
        """Reset message immediately to default."""
        self.current_message = "You're doing great! Keep it up."

    def stop(self):
        """Stop the background worker."""
        self.is_running = False
        self.queue.put(None)
        
    def _worker(self):
        while self.is_running:
            task = self.queue.get()
            if task is None:
                break
                
            action_name, metrics = task
            
            if self.client:
                hour = metrics.get('hour', 12)
                time_of_day = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening" if hour < 22 else "late night"
                
                prompt = f"""
                You are 'FocusMate', an intelligent, empathetic, and slightly witty AI Focus Assistant. 
                Your goal is to gently guide the user back to focus based on their current state.
                
                User Context:
                - Age: {metrics.get('age', 25)} years old
                - Occupation: {metrics.get('occupation', 'student')}
                - Time of day: {hour} ({time_of_day})
                
                Current metrics:
                - Fatigue score: {metrics.get('fatigue_score', 0):.2f} (0=fresh, 1=exhausted)
                - Distracted frames: {metrics.get('consecutive_frames', 0)}
                - Looking away: {'Yes' if abs(metrics.get('head_yaw', 0)) > 30 else 'No'}
                - Phone detected: {'Yes' if metrics.get('phone_count', 0) > 0 else 'No'}
                - User missing: {'Yes' if metrics.get('person_detected', 1) == 0 else 'No'}
                
                The Reinforcement Learning engine just recommended the action: '{action_name}'.
                
                Task: Write a short, natural, conversational response (1-2 sentences) directly to the user.
                - Adapt your tone based on the time of day (e.g., mention it's getting late if it's night) or their fatigue.
                - If they are missing, call them back to the desk.
                - If they are using a phone, tell them to put it away.
                - Do not use quotes around your response.
                """
                try:
                    response = self.client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=prompt
                    )
                    self.current_message = response.text.strip().replace('"', '')
                except Exception as e:
                    self.current_message = f"Please focus on your tasks."
            else:
                # Simulated behavior if no API key
                self.current_message = f"[Simulated LLM] System says: {action_name}."
                
            self.queue.task_done()
