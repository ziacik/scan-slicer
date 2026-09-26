mod detection;
mod editor;
mod openai_detection;
mod ui;

fn main() -> gtk::glib::ExitCode {
    ui::run()
}
