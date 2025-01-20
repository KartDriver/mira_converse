import pygame
import numpy as np
from audio_core import AudioCore

class VolumeWindow:
    def __init__(self, device_name=None):
        try:
            # Initialize PyGame
            pygame.init()
            
            # Window dimensions
            self.width = 300
            self.height = 200
            
            # Initialize window with properties
            try:
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.SHOWN)
                pygame.display.set_caption("Microphone Volume")
            except Exception as e:
                print(f"Could not set window properties: {e}")
            
            # Initialize colors
            self.WHITE = (255, 255, 255)
            self.BLACK = (0, 0, 0)
            self.BLUE = (33, 150, 243)  # Similar to the Qt version's #2196f3
            self.GRAY = (240, 240, 240)
            self.BORDER_GRAY = (204, 204, 204)
            
            # Initialize fonts
            self.title_font = pygame.font.SysFont('Arial', 12)
            self.volume_font = pygame.font.SysFont('Arial', 12, bold=True)
            self.level_font = pygame.font.SysFont('Arial', 11)
            
            # Store device name
            self.device_name = device_name or "No device selected"
            
            # Initialize audio processing
            self.audio_core = AudioCore()
            self.running = True
            self.has_gui = True
            
            # Current volume level
            self.current_volume = 0
            
            print("\nVolume meter window opened successfully")
            
        except Exception as e:
            print(f"\nWarning: Could not create GUI window ({e})")
            print("Volume meter will not be displayed")
            self.running = False
            self.has_gui = False
    
    def process_audio(self, audio_data):
        """Process audio and update volume display"""
        if not self.running or not self.has_gui:
            return
            
        try:
            # Process audio using AudioCore
            result = self.audio_core.process_audio(audio_data)
            
            # Calculate volume using AudioCore's professional metering
            volume = self.audio_core.calculate_volume(audio_data)
            
            # Update volume level (0-100%)
            self.current_volume = int(volume * 100)
            
            # Update display
            self.update()
            
        except Exception:
            pass  # Ignore errors if window is being destroyed
    
    def draw_progress_bar(self, value):
        """Draw the volume progress bar"""
        # Progress bar dimensions
        bar_width = 260
        bar_height = 15
        x = (self.width - bar_width) // 2
        y = 110
        
        # Draw background
        pygame.draw.rect(self.screen, self.GRAY, (x, y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.BORDER_GRAY, (x, y, bar_width, bar_height), 1)
        
        # Draw filled portion
        fill_width = int((value / 100) * (bar_width - 2))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.BLUE, (x + 1, y + 1, fill_width, bar_height - 2))
    
    def update(self):
        """Update the display"""
        if not self.running or not self.has_gui:
            return
            
        try:
            # Handle PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            
            # Clear screen
            self.screen.fill(self.WHITE)
            
            # Draw device name
            device_text = self.title_font.render(self.device_name, True, self.BLACK)
            device_rect = device_text.get_rect(center=(self.width//2, 40))
            self.screen.blit(device_text, device_rect)
            
            # Draw "Volume Level" text
            volume_text = self.volume_font.render("Volume Level", True, self.BLACK)
            volume_rect = volume_text.get_rect(center=(self.width//2, 80))
            self.screen.blit(volume_text, volume_rect)
            
            # Draw progress bar
            self.draw_progress_bar(self.current_volume)
            
            # Draw volume percentage
            level_text = self.level_font.render(f"{self.current_volume}%", True, self.BLACK)
            level_rect = level_text.get_rect(center=(self.width//2, 150))
            self.screen.blit(level_text, level_rect)
            
            # Update display
            pygame.display.flip()
            
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def close(self):
        """Close the window"""
        if self.running and self.has_gui:
            self.running = False
            self.has_gui = False
            try:
                pygame.quit()
            except Exception as e:
                print(f"Error closing window: {e}")
