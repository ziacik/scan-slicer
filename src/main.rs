mod detection;
mod editor;
mod openai_detection;
mod scanner;
mod settings;
mod ui;

fn main() -> gtk::glib::ExitCode {
    ui::run()
}
